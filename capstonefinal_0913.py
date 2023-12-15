# 2023-09-12 final code
from flask import Flask, escape, request, render_template, url_for, redirect, current_app, jsonify, g
from flaskext.mysql import MySQL
from werkzeug.utils import secure_filename
import pymysql
import OpenSSL
from jinja2 import Template
import threading
import os
import librosa
import soundfile as sf
import subprocess
import webbrowser
import signal
from threading import Timer
import asyncio
import datetime
import joblib
import librosa
import wave
from pyaudio import PyAudio, paInt16
import cv2
import mediapipe as mp
import numpy as np
import math
from tensorflow.keras.models import load_model
from collections import Counter # 최종 액션 선택
import time # 시간 측정


mysql = MySQL()
app = Flask(__name__)
# mysql = SQLAlchemy(app)

#MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'hotswkiosk'
app.config['MYSQL_DATABASE_PASSWORD'] = 'hotswkiosk99' 
app.config['MYSQL_DATABASE_DB'] = 'swkiosk'
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'

#mysql = MYSQL(app)
mysql.init_app(app)

# global result
mediapipe_result = None
voice_result = None
yolo_result = None
result = None

restart_requested = False

@app.route('/')
def run_start():
    return render_template('start.html')


@app.route('/runmediapipe', methods = ['GET', 'POST'])
def run_mediapipe():
    global result
    
    model = load_model('./jetson_mediapipe_model_100epc.h5')

    actions = ['elder', 'adult', 'child']
    seq_length = 30

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1)
 
    cap = cv2.VideoCapture(0)

    # 동작 추적 시작 시간을 기록할 변수
    start_time = None

    seq = []
    action_seq = []
    # 최종 동작을 선택하기 위한 윈도우 크기를 초기화합니다.
    window_size = 50  # 예: 50프레임 동안의 동작을 추적
    # 인식 되지 않은 프레임 수 초기화
    mediapipe_none_time = 200

    # 아래 변수를 추가하여 포즈가 인식되지 않은 시간을 추적합니다.
    no_pose_time = 0

    # 3D 각도 계산
    def calculateAngle3D(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
        x1, y1, z1, a1 = landmark1
        x2, y2, z2, a2 = landmark2
        x3, y3, z3, a3 = landmark3

        num = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) + (z2 - z1) * (z3 - z1)

        den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * \
              math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2)

        angle = math.degrees(math.acos(num / den))

        # Check if the angle is less than zero.
        if angle < 0:
            # Add 360 to the found angle.
            angle += 360
            # Return the calculated angle.
        angle = angle / 1000
        return angle

    # 키 구하는 함수
    def calculateHeight(landmark1, landmark2, landmark3):
        x1, y1, z1, a1 = landmark1  # 코
        x2, y2, z2, a2 = landmark2  # 왼쪽발
        x3, y3, z3, a3 = landmark3  # 오른쪽발

        # 왼발과 오른발 사이의 좌표 = pointF
        pointF = ((x2+x3)/2, (y2+y3)/2, (z2+z3)/2)

        # 코(정수리)의 좌표 = pointH
        pointH = (x1, y1, z1)

        # 두 점 사이의 거리 구하기
        a = pointF[1]-pointH[1]
        b = pointF[2]-pointH[2]
        height = math.sqrt(math.pow(a, 2) + math.pow(b, 2))
        height = round(height, 3)
        height = height / 10
        return height

    while cap.isOpened():

        # 시작 시간 기록
        if start_time is None:
            start_time = time.time()

        ret, img = cap.read()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pose.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        x = []
        # joint[0], joint[31], joint[32]의 visibility를 체크하여 포즈 인식을 수행합니다.
        if result.pose_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(img, result.pose_landmarks,     mp_pose.POSE_CONNECTIONS)

            if result.pose_landmarks.landmark[0].visibility > 0.7 and \
                    result.pose_landmarks.landmark[31].visibility > 0.7 and \
                    result.pose_landmarks.landmark[32].visibility > 0.7:

                # 포즈가 인식된 시점에서 no_pose_time을 초기화합니다.
                no_pose_time = 0

                pose_data = []
                joint = np.zeros((33, 4))
                for j, lm in enumerate(result.pose_landmarks.landmark):
                    if j == 0 or (11 <= j <= 32):  # 랜드마크 인덱스 0과 11~32에 해당하는 값만 추가
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        x.append(joint[j][0])
                        x.append(joint[j][1])
                        x.append(joint[j][2])
                        x.append(joint[j][3])

                a = calculateAngle3D(joint[11], joint[23], joint[25])  # (왼쪽: ‘어깨 – 엉덩이 – 무릎’의 각도)
                b = calculateAngle3D(joint[12], joint[24], joint[26])  # (오른쪽: ‘어깨 – 엉덩이 – 무릎’의 각도)
                c = calculateAngle3D(joint[11], joint[23], joint[27])  # (왼쪽: ‘어깨 – 엉덩이 – 발목’의 각도)
                d = calculateAngle3D(joint[12], joint[24], joint[28])  # (왼쪽: ‘어깨 – 엉덩이 – 발목’의 각도)
                e = calculateAngle3D(joint[23], joint[26], joint[25])  # (무릎의 각도: 왼쪽 엉덩이 – 오른쪽 무릎 – 왼쪽 무릎)
                f = calculateAngle3D(joint[24], joint[25], joint[26])  # (무릎의 각도: 오른쪽 엉덩이 – 왼쪽 무릎 – 오른쪽 무릎)

                # 두 점 사이의 비율(X, Y 좌표)(2D) (상하체 길이의 비율)
                g = (math.sqrt(math.pow((joint[11][0] - joint[23][0]), 2) +     math.pow((joint[11][1] - joint[23][1]), 2))) \
                    / (math.sqrt(math.pow((joint[23][0] - joint[27][0]), 2) + math.pow((joint[23][1] - joint[27][1]), 2)))
                g = g / 100
                # g: 2D 왼쪽) (어깨-엉덩이의 길이) / (엉덩이-발목의 길이)
                h = (math.sqrt(
                    math.pow((joint[12][0] - joint[24][0]), 2) +     math.pow((joint[12][1] - joint[24][1]), 2))) \
                    / (math.sqrt(math.pow((joint[24][0] - joint[28][0]), 2) +     math.pow((joint[24][1] - joint[28][1]), 2)))
                h = h / 100
                # h: 2D 오른쪽) (어깨-엉덩이의 길이) / (엉덩이-발목의 길이)

                # 두 점 사이의 비율(X, Y 좌표)(3D)
                i = (math.sqrt(math.pow((joint[11][0] - joint[23][0]), 2)
                               + math.pow((joint[11][1] - joint[23][1]), 2)
                               + math.pow((joint[11][2] - joint[23][2]), 2))) \
                    / (math.sqrt(math.pow((joint[23][0] - joint[27][0]), 2)
                                 + math.pow((joint[23][1] - joint[27][1]), 2)
                                 + math.pow((joint[23][2] - joint[27][2]), 2)))
                i = i / 100
                # i: 3D 왼쪽) (어깨-엉덩이의 길이) / (엉덩이-발목의 길이)
                j = (math.sqrt(math.pow((joint[12][0] - joint[24][0]), 2)
                               + math.pow((joint[12][1] - joint[24][1]), 2)
                               + math.pow((joint[12][2] - joint[24][2]), 2))) \
                    / (math.sqrt(math.pow((joint[24][0] - joint[28][0]), 2)
                                 + math.pow((joint[24][1] - joint[28][1]), 2)
                                 + math.pow((joint[24][2] - joint[28][2]), 2)))
                j = j / 100
                # j: 3D 오른쪽) (어깨-엉덩이의 길이) / (엉덩이-발목의 길이)

                s = calculateHeight(joint[0], joint[31], joint[32])

                data_list = [a, b, c, d, e, f, g, h, i, j, s]
                data = np.array([data_list], dtype=np.float32)
                x_data = np.array([x], dtype=np.float32)
                pose_data = np.concatenate((data, x_data), axis=1)

                # 프레임 당 포즈 데이터 seq에 추가
                seq.append(pose_data)

                # seq 길이 30이라 30까지 데이터 모으기
                if len(seq) < seq_length:
                    continue

                # 모델 예측
                input_data = np.expand_dims(np.array(seq[-seq_length:],     dtype=np.float32), axis=0)
                input_data = np.squeeze(input_data, axis=2)
                y_pred = model.predict(input_data).squeeze()
                # y_pred: 노인, 청장년층, 어린이 중 최대 확률을 가지는 포즈 클래스의 인덱스를 찾기 위한 배열

                i_pred = int(np.argmax(y_pred))
                # i_pred: 가장 높은 확률을 갖는 포즈 클래스의 인덱스를 도출(0: elder, 1: adult, 2; younger)
                conf = y_pred[i_pred]
                # conf: 클래스가 정답일 확률(0.xx 형태)

                if conf < 0.9:
                # 정확도가 90% 이하일 때는 수행 지속함
                    continue

                action = actions[i_pred] # 클래스의 레이블을 가져옴. action에는 elder, adult, younger 등이 나타남
                action_seq.append(action) # 도출된 action을 action_seq 리스트에 추가함. 최근 프레임 동안 인식된 동작 추정

                if len(action_seq) < 3:
                    continue
                #############
                if action == 'elder':
                    result = 'elder'
                    time.sleep(5)
                    print("Final Result: ", result)
                    end_time = time.time()  # 끝 시간 기록
                    elapsed_time = end_time - start_time
                    print(f"Time taken to detect final result: {elapsed_time} seconds")
                    break
                    

                # action_seq 리스트의 길이가 윈도우 크기보다 큰 경우,
                if len(action_seq) > window_size:
                    # action_history 리스트의 첫 번째 요소를 제거하여 윈도우 크기를 유지합니다.
                    action_seq.pop(0)
                    # action_history 리스트에서 가장 많이 나타난 action을 찾습니다.
                    most_common_action = Counter(action_seq).most_common(1)
                    # most_common_action은 리스트이므로 첫 번째 요소를 선택합니다.
                    result = most_common_action[0][0]
                    # 최종 동작을 출력하거나 사용할 수 있습니다.
                    print("Final Result:", result)


                    # 최종 동작이 나타나면 시간을 출력하고 루프를 종료
                    if result is not None:
                        end_time = time.time()  # 끝 시간 기록
                        elapsed_time = end_time - start_time
                        print(f"Time taken to detect final result: {elapsed_time} seconds")
                        break

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action
                print("Now Pose Action:", this_action)

                # 정답 그리기 코드
                cv2.putText(img, f'{this_action.upper()}',     org=(int(result.pose_landmarks.landmark[0].x * img.shape[1]),
                                                                int(result.pose_landmarks.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,     color=(255, 255, 255), thickness=2)


            else:
            # 포즈 인식 실패한 경우
            # 정답 그리기 코드(정답 도출이 안 된 경우 ?로 표시)
                this_action = '?'
                cv2.putText(img, f'{this_action.upper()}',     org=(int(result.pose_landmarks.landmark[0].x * img.shape[1]+100),
                                                                int(result.pose_landmarks.landmark[0].y * img.shape[0]-40)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,     color=(255, 255, 255), thickness=4)

                # no_pose_time을 증가시킵니다.
                no_pose_time += 1

                # no_pose_time이 none_time 이상이면 'adult'로 자동 인식합니다.
                if no_pose_time >= mediapipe_none_time:
                    most_common_action = 'adult'
                    result = most_common_action
                    print("No pose detected. Auto detected as 'adult'")
                    break


        cv2.imshow('Filter', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return result



def run_voice():

    global result
   
    model = load_model("./lstm_voice_jetson.h5")
    label_map = {0: 'elder', 1: 'child', 2: 'adult'}  


    def extract_features(audio_path):
        X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast',         duration=2.5, sr=44100, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=np.array(sample_rate), n_mfcc=13), axis=0)
   
        X_train_shape = model.input_shape[1]
        mfccs = np.pad(mfccs, (0, X_train_shape - len(mfccs)), mode='constant')
        return mfccs


    def my_record(filename):
        NUM_SAMPLES = 2000
        framerate = 16000
        channels = 1
        sampwidth = 2
        TIME = 10

        def save_wave_file(data):
            wf = wave.open(filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(b"".join(data))
            wf.close()

        pa = PyAudio()
        stream = pa.open(format=paInt16, channels=1,
                         rate=framerate, input=True,
                         frames_per_buffer=NUM_SAMPLES)
        my_buf = []
        count = 0
        time.sleep(1)
        print('Start your record')
        while count < TIME*3:
            string_audio_data = stream.read(NUM_SAMPLES)
            my_buf.append(string_audio_data)
            count += 1
            print('.')
        print('Done!')
        save_wave_file(my_buf)
        stream.close()


    new_audio_path = './recorded_audio.wav'
    my_record(new_audio_path)


    new_features = extract_features(new_audio_path)


    new_features = new_features.reshape(1, model.input_shape[1], 1)


    predicted_probabilities = model.predict(new_features)[0]
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class = label_map.get(predicted_class_index)


    print(predicted_probabilities)
    print("Predicted Class(elder/child/adult):", predicted_class)
    result = predicted_class
    return result



@app.route('/mediapiperesult')
def finalmediapiperesult():
     global result
     return jsonify({'result': str(result)})





@app.route('/isPickup')
def isPickup():
    return render_template('isPickup.html') 


@app.route('/end')
def end():
    return render_template('end.html')

@app.route('/button')
def button():
    return render_template('button.html')


@app.route('/run_yolo')
def run_yolo():
    global result
    num_detections = 0  # Number of detections
    max_detections = 30 # Maximum number of detections within 30 seconds
    has_wheelchair = False  # Flag for detecting wheelchair class
    has_white_cane = False  # Flag for detecting white cane class
    #frame_index=0

    # YOLOv4-tiny 모델을 위한 설정
    model_config = 'yolov3tiny-modify.cfg'
    model_weights = 'yolov3tiny-modify_best.weights'
    classes_file = 'obj.names'

    # 클래스 이름 로드
    with open(classes_file, 'r') as f:
        classes = f.read().splitlines()

    # YOLO 네트워크 불러오기
    net = cv2.dnn.readNet(model_weights, model_config)

    
    # GPU 사용 (Jetson Nano에 적합한 설정)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 비디오 캡처 설정
    cap = cv2.VideoCapture('/dev/video0')
    start_time = time.time()  # 시작 시간 기록

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        print("Yolo cap is opened")

        # 입력 이미지를 YOLO 입력 형식으로 변환
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # YOLO 객체 탐지 수행
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)

        # 탐지된 객체 정보 처리
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # 탐지된 객체의 경계상자 좌표 계산
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # 비최대 억제를 사용하여 중복 탐지 제거
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # 탐지된 객체를 화면에 표시
        font_scale = 0.6
        thickness = 1
        for i in indices:
            i = i[0]
            box = boxes[i]
            left, top, width, height = box
            class_id = class_ids[i]
            label = f'{classes[class_id]}'

            if class_id == 2:
                has_wheelchair = True
            elif class_id == 3:
                has_white_cane = True

            # 객체 경계 상자 및 클래스 레이블 그리기
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), thickness)
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # 화면에 출력
        cv2.imshow('Object Detection', frame)

        # Increment the number of detections
        num_detections += 1

        # Check if the maximum number of detections has been reached
        if num_detections >= max_detections:
            cap.release()
            cv2.destroyWindow('Object Detection')
            break

        # Check if 30 seconds have passed
        if time.time() - start_time > 50:
            print("Yolo 30s timeout")
            cap.release()
            cv2.destroyWindow('Object Detection')
            break

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) == ord('q'):
            break

    # Determine the final result based on the detected classes
    if has_wheelchair:
        result = 'wheelchair'

        
    elif has_white_cane:
        result = 'canes'

    else:
        #result = 'person'
        result = 'person'
        

    print("Yolo_result", result)
          

          
@app.route('/runrun_all', methods = ['GET', 'POST'])
def runrun_all():
    global result
    run_yolo()
    print("yolo is done")

    if (result == "wheelchair"):
        print("voice start")
        time.sleep(1)
        run_voice()
        print("voice is done")
        sub = result
        if(sub == "elder"):
            result = "wheelelder"            
        elif(sub == "child"):
            result = "wheelchild"           
        else:
            result = "wheel"

        print("voice thread", result)

    elif (result == "canes"):
         
        print("voice start")
        time.sleep(1)
        run_voice()
        print("voice is done")
        sub = result
        if(sub == "elder"):
            result = "caneelder"
        elif(sub == "child"):
            result = "canechild"
        else:
            result = "caneadult"   

        print("voice thread", result) 
    else:
        print("mediapipe start")
        run_mediapipe()
        print("mediapipe is done")
        print("mediapipe thread", result)

    print("Final Done!!")
    return result


@app.route('/wait', methods = ['GET', 'POST'])
def wait():
    return render_template('wait.html')

@app.route('/second', methods = ['GET', 'POST'])
def second():
    return render_template('second.html')          


@app.route('/restart', methods = ['GET', 'POST'])
def start():
    result = None
    global restart_requested
    restart_requested = True

    return render_template('restart.html')     



@app.route("/elderstart", methods=["GET", "POST"])
def elderstart():
    return render_template("elderstart.html")


@app.route("/canestart", methods=["GET", "POST"])
def canestart():
    return render_template("canestart.html")


@app.route("/wheelelderstart", methods=["GET", "POST"])
def wheelelderstart():
    return render_template("wheelelderstart.html")


@app.route("/wheelstart", methods=["GET", "POST"])
def wheelstart():
    return render_template("wheelstart.html")


@app.route("/caneelderstart", methods=["GET", "POST"])
def caneelderstart():
    return render_template("caneelderstart.html")


@app.route("/childstart", methods=["GET", "POST"])
def childstart():
    return render_template("childstart.html")


@app.route("/canechildstart", methods=["GET", "POST"])
def canechildstart():
    return render_template("canechildstart.html")


@app.route("/wheelchildstart", methods=["GET", "POST"])
def wheelchildstart():
    return render_template("wheelchildstart.html")

@app.route("/caneisPickup", methods=["GET", "POST"])
def caneisPickup():
    return render_template("caneisPickup.html")


@app.route("/caneend", methods=["GET", "POST"])
def caneend():
    return render_template("caneend.html")


@app.route('/mainpage', methods = ['GET', 'POST'])
def view():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)
    sql = "SELECT img FROM menu WHERE usertype LIKE '%person%'"  # 실행할 SQL문
    cursor.execute(sql)  # 메소드로 전달해 명령문을 실행
    data = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'";  # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql4 = "SELECT menuname FROM menu WHERE usertype LIKE '%person%'" 
    cursor.execute(sql4)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    menu_list = []
    for menu in menudata:
        menu_list.append(menu[0])



    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%person%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7) 
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])



  
    data_list = []
    data_new_list=[]
    data_bowl_list=[]
    data_series_list=[]
   
 

 
    for obj in data:
        data_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataNew:
        data_new_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서
    
    for obj in dataSeries:
        data_series_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서




    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)
 
    return render_template('person.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)
  # html을 렌더하며 DB에서 받아온 값들을   넘김

@app.route('/elder', methods = ['GET', 'POST'])
def elder():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)

    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'";  # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄



    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%person%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7) 
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])

 

    sql8 = 'SELECT img FROM menu WHERE usertype LIKE "%elder%"'   # 실행할 SQL문
    cursor.execute(sql8)  # 메소드로 전달해 명령문을 실행
    dataelder = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql9 = "SELECT menuname FROM menu WHERE usertype LIKE '%elder%'" 
    cursor.execute(sql9)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    elder_menu_list = []
    for menu in menudata:
        elder_menu_list.append(menu[0])

    sql10 = "SELECT money FROM menu WHERE usertype LIKE '%elder%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql10) 
    moneymenudata = cursor.fetchall()
    elder_money_menu_list = []
    for money in moneymenudata:
        elder_money_menu_list.append(money[0])

   



    
  

    data_new_list=[]
    data_bowl_list=[]
    data_series_list=[]
    data_elder_list=[]
  
 

 
   

    for obj in dataNew:
        data_new_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서
    
    for obj in dataSeries:
        data_series_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataelder:
        data_elder_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서



    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)
 
    return render_template('elder.html',  data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, all_menu_list=all_menu_list , data_elder_list=data_elder_list
                           ,elder_menu_list=elder_menu_list, elder_money_menu_list=elder_money_menu_list,money_menu_list=money_menu_list)

  # html을 렌더하며 DB에서 받아온 값들을 넘김


@app.route('/cane', methods = ['GET', 'POST'])
def cane():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)
 

    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'";  # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

   


    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%person%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7) 
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])

   

    sql11 = sql = "SELECT img FROM menu WHERE usertype LIKE '%blind%'"  # 실행할 SQL문
    cursor.execute(sql11)  # 메소드로 전달해 명령문을 실행
    datacane = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄


    sql12 = "SELECT menuname FROM menu WHERE usertype LIKE '%blind%'" 
    cursor.execute(sql12)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    cane_menu_list = []
    for menu in menudata:
        cane_menu_list.append(menu[0])

    sql13 = "SELECT money FROM menu WHERE usertype LIKE '%blind%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql13) 
    moneymenudata = cursor.fetchall()
    cane_money_menu_list = []
    for money in moneymenudata:
        cane_money_menu_list.append(money[0])

  
    
    data_new_list=[]
    data_bowl_list=[]
    data_series_list=[]
    data_cane_list=[]
  
 

 
   
    for obj in dataNew:
        data_new_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서
    
    for obj in dataSeries:
        data_series_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서


    for obj in datacane:
        data_cane_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서




    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)
 
    return render_template('cane.html',  data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list, 
                         cane_menu_list=cane_menu_list, cane_money_menu_list=cane_money_menu_list, data_cane_list=data_cane_list,)
  # html을 렌더하며 DB에서 받아온 값들을 넘김


@app.route("/canechild", methods=["GET", "POST"])
def canechild():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)

    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'"
    # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql4 = "SELECT menuname FROM menu WHERE usertype LIKE '%blind%'"
    cursor.execute(sql4)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    cane_menu_list = []
    for menu in menudata:
        cane_menu_list.append(menu[0])

    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%blind%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7)
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])

    sql20 = "SELECT img FROM menu WHERE usertype LIKE '%blind%' AND usertype LIKE '%child%';"  # 실행할 SQL문
    cursor.execute(sql20)  # 메소드로 전달해 명령문을 실행
    datacanechild = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql21 = "SELECT menuname FROM menu WHERE usertype LIKE '%blind%' AND usertype LIKE '%child%'"
    cursor.execute(sql21)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    cane_child_menu_list = []
    for menu in menudata:
        cane_child_menu_list.append(menu[0])

    sql22 = "SELECT money FROM menu WHERE usertype LIKE '%blind%' AND usertype LIKE '%child%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql22)
    moneymenudata = cursor.fetchall()
    cane_child_money_menu_list = []
    for money in moneymenudata:
        cane_child_money_menu_list.append(money[0])

    data_new_list = []
    data_bowl_list = []
    data_series_list = []
    data_cane_child_list = []

    for obj in dataNew:
        data_new_list.append("all/" + obj[0] + ".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/" + obj[0] + ".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataSeries:
        data_series_list.append("all/" + obj[0] + ".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in datacanechild:
        data_cane_child_list.append("all/" + obj[0] + ".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)

    return render_template(
        "canechild.html",
        data_cane_child_list=data_cane_child_list,
        data_new_list=data_new_list,
        data_bowl_list=data_bowl_list,
        data_series_list=data_series_list,
        cane_menu_list=cane_menu_list,
        all_menu_list=all_menu_list,
        money_menu_list=money_menu_list,
        cane_child_menu_list=cane_child_menu_list,
        cane_child_money_menu_list=cane_child_money_menu_list,
    )


# html을 렌더하며 DB에서 받아온 값들을 넘김


@app.route("/caneelder", methods=["GET", "POST"])
def caneelder():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)

    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'"
    # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%person%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7)
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])

    sql8 = 'SELECT img FROM menu WHERE usertype LIKE "%elder%"'  # 실행할 SQL문
    cursor.execute(sql8)  # 메소드로 전달해 명령문을 실행
    dataelder = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql9 = "SELECT menuname FROM menu WHERE usertype LIKE '%elder%'"
    cursor.execute(sql9)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    elder_menu_list = []
    for menu in menudata:
        elder_menu_list.append(menu[0])

    sql10 = "SELECT money FROM menu WHERE usertype LIKE '%elder%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql10)
    moneymenudata = cursor.fetchall()
    elder_money_menu_list = []
    for money in moneymenudata:
        elder_money_menu_list.append(money[0])

    data_new_list = []
    data_bowl_list = []
    data_series_list = []
    data_elder_list = []

    for obj in dataNew:
        data_new_list.append("all/" + obj[0] + ".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/" + obj[0] + ".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataSeries:
        data_series_list.append("all/" + obj[0] + ".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataelder:
        data_elder_list.append("all/" + obj[0] + ".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)

    return render_template(
        "caneelder.html",
        data_new_list=data_new_list,
        data_bowl_list=data_bowl_list,
        data_series_list=data_series_list,
        all_menu_list=all_menu_list,
        data_elder_list=data_elder_list,
        elder_menu_list=elder_menu_list,
        elder_money_menu_list=elder_money_menu_list,
        money_menu_list=money_menu_list,
    )


# html을 렌더하며 DB에서 받아온 값들을 넘김





@app.route('/child', methods = ['GET', 'POST'])
def child():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)
 
    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'";  # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄



    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%person%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7) 
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])

    sql17 = sql = "SELECT img FROM menu WHERE usertype LIKE '%child%'"  # 실행할 SQL문
    cursor.execute(sql17)  # 메소드로 전달해 명령문을 실행
    datachild = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄


    sql18 = "SELECT menuname FROM menu WHERE usertype LIKE '%child%'" 
    cursor.execute(sql18 )  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    child_menu_list = []
    for menu in menudata:
        child_menu_list.append(menu[0])

    sql19 = "SELECT money FROM menu WHERE usertype LIKE '%child%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql19) 
    moneymenudata = cursor.fetchall()
    child_money_menu_list = []
    for money in moneymenudata:
        child_money_menu_list.append(money[0])

    
  

    data_new_list=[]
    data_bowl_list=[]
    data_series_list=[]
 
    data_child_list=[]


 

    for obj in dataNew:
        data_new_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서
    
    for obj in dataSeries:
        data_series_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서


    for obj in datachild:
        data_child_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서
    



    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)
 
    return render_template('child.html' ,data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list,
                           child_menu_list=child_menu_list, child_money_menu_list=child_money_menu_list, data_child_list=data_child_list)
  # html을 렌더하며 DB에서 받아온 값들을 넘김



@app.route('/wheel', methods = ['GET', 'POST'])
def wheel():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)


    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'";  # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql4 = "SELECT menuname FROM menu WHERE usertype LIKE '%person%'" 
    cursor.execute(sql4)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    menu_list = []
    for menu in menudata:
        menu_list.append(menu[0])



    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%person%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7) 
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])

   
    sql14 = sql = "SELECT img FROM menu WHERE usertype LIKE '%wheel%'"  # 실행할 SQL문
    cursor.execute(sql14)  # 메소드로 전달해 명령문을 실행
    datawheel = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄


    sql15 = "SELECT menuname FROM menu WHERE usertype LIKE '%wheel%'" 
    cursor.execute(sql15)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    wheel_menu_list = []
    for menu in menudata:
        wheel_menu_list.append(menu[0])

    sql16 = "SELECT money FROM menu WHERE usertype LIKE '%wheel%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql16) 
    moneymenudata = cursor.fetchall()
    wheel_money_menu_list = []
    for money in moneymenudata:
        wheel_money_menu_list.append(money[0])


    data_new_list=[]
    data_bowl_list=[]
    data_series_list=[]
    data_wheel_list=[]
  


    for obj in dataNew:
        data_new_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서
    
    for obj in dataSeries:
        data_series_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서


    for obj in datawheel:
        data_wheel_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

  


    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)
 
    return render_template('wheel.html',  data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list
                           ,wheel_menu_list=wheel_menu_list, wheel_money_menu_list=wheel_money_menu_list, data_wheel_list=data_wheel_list)
  # html을 렌더하며 DB에서 받아온 값들을 넘김



@app.route('/wheelelder', methods = ['GET', 'POST'])
def wheelelder():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)
    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'";  # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql4 = "SELECT menuname FROM menu WHERE usertype LIKE '%person%'" 
    cursor.execute(sql4)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    menu_list = []
    for menu in menudata:
        menu_list.append(menu[0])



    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%person%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7) 
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])

   
    sql26 = sql = "SELECT img FROM menu WHERE usertype LIKE '%wheel%' AND usertype LIKE '%elder%';"  # 실행할 SQL문
    cursor.execute(sql26)  # 메소드로 전달해 명령문을 실행
    datawheelelder = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄


    sql27 = "SELECT menuname FROM menu WHERE usertype LIKE '%wheel%' AND usertype LIKE '%elder%'" 
    cursor.execute(sql27 )  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    wheel_elder_menu_list = []
    for menu in menudata:
        wheel_elder_menu_list.append(menu[0])

    sql28 = "SELECT money FROM menu WHERE usertype LIKE '%wheel%' AND usertype LIKE '%elder%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql28) 
    moneymenudata = cursor.fetchall()
    wheel_elder_money_menu_list = []
    for money in moneymenudata:
        wheel_elder_money_menu_list.append(money[0])






    
 
    data_new_list=[]
    data_bowl_list=[]
    data_series_list=[]
    data_wheel_elder_list=[]

 

 

    for obj in dataNew:
        data_new_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서
    
    for obj in dataSeries:
        data_series_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서


    for obj in datawheelelder:
        data_wheel_elder_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서



    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)
 
    return render_template('wheelelder.html', data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list
                           ,wheel_elder_menu_list=wheel_elder_menu_list, wheel_elder_money_menu_list=wheel_elder_money_menu_list, data_wheel_elder_list=data_wheel_elder_list)
  # html을 렌더하며 DB에서 받아온 값들을 넘김




@app.route('/wheelchild', methods = ['GET', 'POST'])
def wheelchild():
    conn = mysql.connect()  # DB와 연결
    cursor = conn.cursor()  # connection으로부터 cursor 생성 (데이터베이스의 Fetch 관리)
    sql1 = "SELECT img FROM menu WHERE category ='신메뉴'";  # 실행할 SQL문
    cursor.execute(sql1)  # 메소드로 전달해 명령문을 실행
    dataNew = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql2 = "SELECT img FROM menu WHERE category ='정식시리즈'"  # 실행할 SQL문
    cursor.execute(sql2)  # 메소드로 전달해 명령문을 실행
    dataSeries = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql3 = "SELECT img FROM menu WHERE category ='덮밥'"  # 실행할 SQL문
    cursor.execute(sql3)  # 메소드로 전달해 명령문을 실행
    dataBowl = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄

    sql4 = "SELECT menuname FROM menu WHERE usertype LIKE '%person%'" 
    cursor.execute(sql4)  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    menu_list = []
    for menu in menudata:
        menu_list.append(menu[0])



    # sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴'"
    # cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    # newmenudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    # new_menu_list = []
    # for menu in newmenudata:
    #   new_menu_list.append(menu[0])

    sql5 = "SELECT menuname FROM menu WHERE category ='신메뉴' OR category ='덮밥' OR category = '정식시리즈';"
    cursor.execute(sql5)  # 메소드로 전달해 명령문을 실행
    allmenudata = cursor.fetchall()
    all_menu_list = []
    for menu in allmenudata:
        all_menu_list.append(menu[0])

    sql7 = "SELECT money FROM menu WHERE usertype LIKE '%person%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql7) 
    moneymenudata = cursor.fetchall()
    money_menu_list = []
    for money in moneymenudata:
        money_menu_list.append(money[0])

   
    sql26 = sql = "SELECT img FROM menu WHERE usertype LIKE '%wheel%' AND usertype LIKE '%child%';"  # 실행할 SQL문
    cursor.execute(sql26)  # 메소드로 전달해 명령문을 실행
    datawheelelder = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄


    sql27 = "SELECT menuname FROM menu WHERE usertype LIKE '%wheel%' AND usertype LIKE '%child%'" 
    cursor.execute(sql27 )  # 메소드로 전달해 명령문을 실행
    menudata = cursor.fetchall()  # 실행한 결과 데이터를 꺼냄
    wheel_child_menu_list = []
    for menu in menudata:
        wheel_child_menu_list.append(menu[0])

    sql28 = "SELECT money FROM menu WHERE usertype LIKE '%wheel%' AND usertype LIKE '%child%' OR category = '신메뉴' OR category = '덮밥' OR category = '정식시리즈'"
    cursor.execute(sql28) 
    moneymenudata = cursor.fetchall()
    wheel_child_money_menu_list = []
    for money in moneymenudata:
        wheel_child_money_menu_list.append(money[0])






    
 
    data_new_list=[]
    data_bowl_list=[]
    data_series_list=[]
    data_wheel_child_list=[]

 

 

    for obj in dataNew:
        data_new_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서

    for obj in dataBowl:
        data_bowl_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서
    
    for obj in dataSeries:
        data_series_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서


    for obj in datawheelelder:
        data_wheel_child_list.append("all/"+obj[0]+".jpg")  # 튜플 안의 데이터를 하나씩 조회해서



    cursor.close()
    conn.close()

    # result = main_program()
    # if result == 'youth':
    #     return render_template('capstoneyouth.html')
    # else:
    #     return render_template('capstone2.html', data_list=data_list, data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list)
 
    return render_template('wheelchild.html', data_new_list=data_new_list, data_bowl_list=data_bowl_list, data_series_list=data_series_list, menu_list=menu_list, all_menu_list=all_menu_list ,money_menu_list=money_menu_list
                           ,wheel_child_menu_list=wheel_child_menu_list, wheel_child_money_menu_list=wheel_child_money_menu_list, data_wheel_child_list=data_wheel_child_list)
  # html을 렌더하며 DB에서 받아온 값들을 넘김






def start_flask():
    flask_thread = threading.Thread(target=app.run, kwargs={'host': '127.0.0.1', 'port': 5000})
    flask_thread.start()
    
if __name__ == "__main__":

    webbrowser.open('http://127.0.0.1:5000')
    start_flask()

    # Run runrun_all() once after restart is requested
    time.sleep(2)
    runrun_all()

    
    while True:
        if restart_requested:
            result = None
            time.sleep(4)
            runrun_all()
            restart_requested = False  # Reset the flag

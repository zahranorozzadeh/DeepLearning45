import cv2
import telebot
import numpy as np
from keras.models import load_model
from retinaface import RetinaFace
import matplotlib.pyplot as plt


model = load_model("/content/drive/MyDrive/gender_model.h5")
bot = telebot.TeleBot("1998425031:AAEO7yJO_KGEbwiiPjo-ZzDfY-OoS1sbhtI")

width = 224
height = 224

@bot.message_handler(commands=['start'])
def say_hello(messages):
    bot.send_message(messages.chat.id, f'Wellcome Dear {messages.from_user.first_name}🌹')
    bot.send_message(messages.chat.id, f'Here you can distinguish man from woman')
    bot.send_message(messages.chat.id, f'Now send me the photo so I can tell you😉')


@bot.message_handler(content_types=['photo'])
def photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = file_info.file_path
    with open("mohtava.jpg" , 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.send_message(message.chat.id, 'Processing...\nPlease Wait')

    face = RetinaFace.extract_faces(img_path = "mohtava.jpg", align = True)
    if len(face) > 0:
        face = cv2.cvtColor(face[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite("mohtava.jpg", face)
        image = cv2.imread("mohtava.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image = image/255
        image = image.reshape(1, width, height, 3)
        pred = model.predict([image])

        res = np.argmax(pred)
        if res == 0:
          bot.reply_to(message, 'man')
        else:
          bot.reply_to(message, 'woman')
    else:
        bot.send_message(message.chat.id, 'No man or woman')


bot.polling()
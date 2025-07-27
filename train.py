# -*- coding: utf-8 -*-

#  *原训练代码*
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
# setting.json "D:\\YOLOV11\\YOLOV11(datasets)\\yolov11\\datasets"
# setting.yaml D:\YOLOV11\YOLOV11(datasets)\yolov11\datasets
if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'D:\YOLOV11\YOLOV11(datasets)\yolov11\ultralytics\cfg\models\11\TiaoZheng\10.yaml')
    result = model.train(data=r'D:\YOLOV11\YOLOV11(datasets)\yolov11\ultralytics\cfg\datasets\URPC2020-split.yaml',
                imgsz=640,
                epochs=500,
                batch=4,
                workers=0,
                device=0,
                #optimizer='SGD', #SGD在每次更新时仅使用一个 batch样本来计算梯度。更快的计算速度，更好的泛化能力。
                #close_mosaic=10, #10个 epochs后，停止使用 Mosaic 数据增强。在训练初期使用 Mosaic 数据增强，可以增加训练样本的多样性，从而提高模型的泛化能力；而随着训练的进行，停止这种增强有助于模型集中学习特定物体的特征。
                resume=False,
                project='runs/URPC2020-split',
                name='10-500epochs-1-quan',
                single_cls=False,
                cache=False,
                )




#import os
#dataset_path = "D:/YOLOV11/YOLOV11(datasets)/yolov11/datasets/URPC2018/images/valid"
#dataset_path2 = "D:/YOLOV11/YOLOV11(datasets)/yolov11/datasets/URPC2018/images/train"
#print("Checking dataset path:", os.path.exists(dataset_path))
#print("Checking dataset path:", os.path.exists(dataset_path2))

#
# from ultralytics import YOLO
# import warnings
#
# warnings.filterwarnings('ignore')
# from pathlib import Path
#
# if __name__ == '__main__':
#     # 加载模型
#     model = YOLO("ultralytics/cfg/11/yolo11.yaml")  # 你要选择的模型yaml文件地址
#     # Use the model
#     results = model.train(data=r"你的数据集的yaml文件地址",
#                           epochs=100, batch=16, imgsz=640, workers=4, name=Path(model.cfg).stem)  # 训练模型
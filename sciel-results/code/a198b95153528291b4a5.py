import numpy as np
import time

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Build components
    U222, V222, W222 = _build_strassen_2x2x2()
    U122, V122, W122 = _build_schoolbook(1, 2, 2)
    U212, V212, W212 = _build_schoolbook(2, 1, 2)
    U132, V132, W132 = _build_schoolbook(1, 3, 2)
    
    # M_{3,2,2} = M_{2,2,2} + M_{1,2,2} (row split)
    r = len(U222[0]) + len(U122[0])
    U322 = [[0]*r for _ in range(6)]
    V322 = [[0]*r for _ in range(4)]
    W322 = [[0]*r for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U322[i*2+j][c] = U222[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V322[j*2+k][c] = V222[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W322[k*2+i][c] = W222[k*2+i][c]
    for i in range(1):
        for j in range(2):
            for c in range(4):
                U322[(2+i)*2+j][7+c] = U122[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(4):
                V322[j*2+k][7+c] = V122[j*2+k][c]
    for k in range(2):
        for i in range(1):
            for c in range(4):
                W322[k*3+i][7+c] = W122[k*1+i][c]
    
    # M_{2,3,2} = M_{2,2,2} + M_{2,1,2} (col split)
    r = len(U222[0]) + len(U212[0])
    U232 = [[0]*r for _ in range(6)]
    V232 = [[0]*r for _ in range(6)]
    W232 = [[0]*r for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U232[i*3+j][c] = U222[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V232[j*2+k][c] = V222[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W232[k*2+i][c] = W222[k*2+i][c]
    for i in range(2):
        for j in range(1):
            for c in range(4):
                U232[i*3+(2+j)][7+c] = U212[i*1+j][c]
    for j in range(1):
        for k in range(2):
            for c in range(4):
                V232[(2+j)*2+k][7+c] = V212[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(4):
                W232[k*2+i][7+c] = W212[k*2+i][c]
    
    # M_{3,3,2} = M_{2,3,2} + M_{1,3,2} (row split)
    r = len(U232[0]) + len(U132[0])
    U332 = [[0]*r for _ in range(9)]
    V332 = [[0]*r for _ in range(6)]
    W332 = [[0]*r for _ in range(6)]
    for i in range(2):
        for j in range(3):
            for c in range(11):
                U332[i*3+j][c] = U232[i*3+j][c]
    for j in range(3):
        for k in range(2):
            for c in range(11):
                V332[j*2+k][c] = V232[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(11):
                W332[k*2+i][c] = W232[k*2+i][c]
    for i in range(1):
        for j in range(3):
            for c in range(6):
                U332[(2+i)*3+j][11+c] = U132[i*3+j][c]
    for j in range(3):
        for k in range(2):
            for c in range(6):
                V332[j*2+k][11+c] = V132[j*3+k][c]
    for k in range(2):
        for i in range(1):
            for c in range(6):
                W332[k*3+i][11+c] = W132[k*1+i][c]
    
    # M_{3,5,2} = M_{3,2,2} + M_{3,3,2} (col split)
    r = len(U322[0]) + len(U332[0])
    U352 = [[0]*r for _ in range(15)]
    V352 = [[0]*r for _ in range(10)]
    W352 = [[0]*r for _ in range(6)]
    for i in range(3):
        for j in range(2):
            for c in range(11):
                U352[i*5+j][c] = U322[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(11):
                V352[j*2+k][c] = V322[j*2+k][c]
    for k in range(2):
        for i in range(3):
            for c in range(11):
                W352[k*3+i][c] = W322[k*3+i][c]
    for i in range(3):
        for j in range(3):
            for c in range(17):
                U352[i*5+(2+j)][11+c] = U332[i*3+j][c]
    for j in range(3):
        for k in range(2):
            for c in range(17):
                V352[(2+j)*2+k][11+c] = V332[j*2+k][c]
    for k in range(2):
        for i in range(3):
            for c in range(17):
                W352[k*3+i][11+c] = W332[k*3+i][c]
    
    # M_{3,2,4} = M_{3,2,2} (k=0,1) + M_{3,2,2} (k=2,3)
    r = 11 + 11
    U324 = [[0]*r for _ in range(6)]
    V324 = [[0]*r for _ in range(8)]
    W324 = [[0]*r for _ in range(12)]
    for i in range(3):
        for j in range(2):
            for c in range(11):
                U324[i*2+j][c] = U322[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(11):
                V324[j*4+k][c] = V322[j*2+k][c]
    for k in range(2):
        for i in range(3):
            for c in range(11):
                W324[k*3+i][c] = W322[k*3+i][c]
    for i in range(3):
        for j in range(2):
            for c in range(11):
                U324[i*2+j][11+c] = U322[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(11):
                V324[j*4+(2+k)][11+c] = V322[j*2+k][c]
    for k in range(2):
        for i in range(3):
            for c in range(11):
                W324[(2+k)*3+i][11+c] = W322[k*3+i][c]
    
    # M_{3,3,4} = M_{3,3,2} (k=0,1) + M_{3,3,2} (k=2,3)
    r = 17 + 17
    U334 = [[0]*r for _ in range(9)]
    V334 = [[0]*r for _ in range(12)]
    W334 = [[0]*r for _ in range(12)]
    for i in range(3):
        for j in range(3):
            for c in range(17):
                U334[i*3+j][c] = U332[i*3+j][c]
    for j in range(3):
        for k in range(2):
            for c in range(17):
                V334[j*4+k][c] = V332[j*2+k][c]
    for k in range(2):
        for i in range(3):
            for c in range(17):
                W334[k*3+i][c] = W332[k*3+i][c]
    for i in range(3):
        for j in range(3):
            for c in range(17):
                U334[i*3+j][17+c] = U332[i*3+j][c]
    for j in range(3):
        for k in range(2):
            for c in range(17):
                V334[j*4+(2+k)][17+c] = V332[j*2+k][c]
    for k in range(2):
        for i in range(3):
            for c in range(17):
                W334[(2+k)*3+i][17+c] = W332[k*3+i][c]
    
    # M_{3,5,4} = M_{3,2,4} + M_{3,3,4} (col split)
    r = 22 + 34
    U354 = [[0]*r for _ in range(15)]
    V354 = [[0]*r for _ in range(20)]
    W354 = [[0]*r for _ in range(12)]
    for i in range(3):
        for j in range(2):
            for c in range(22):
                U354[i*5+j][c] = U324[i*2+j][c]
    for j in range(2):
        for k in range(4):
            for c in range(22):
                V354[j*4+k][c] = V324[j*4+k][c]
    for k in range(4):
        for i in range(3):
            for c in range(22):
                W354[k*3+i][c] = W324[k*3+i][c]
    for i in range(3):
        for j in range(3):
            for c in range(34):
                U354[i*5+(2+j)][22+c] = U334[i*3+j][c]
    for j in range(3):
        for k in range(4):
            for c in range(34):
                V354[(2+j)*4+k][22+c] = V334[j*4+k][c]
    for k in range(4):
        for i in range(3):
            for c in range(34):
                W354[k*3+i][22+c] = W334[k*3+i][c]
    
    # M_{3,5,6} = M_{3,5,2} + M_{3,5,4} (col split in p)
    r = 28 + 56
    U = [[0]*r for _ in range(15)]
    V = [[0]*r for _ in range(30)]
    W = [[0]*r for _ in range(18)]
    for i in range(3):
        for j in range(5):
            for c in range(28):
                U[i*5+j][c] = U352[i*5+j][c]
    for j in range(5):
        for k in range(2):
            for c in range(28):
                V[j*6+k][c] = V352[j*2+k][c]
    for k in range(2):
        for i in range(3):
            for c in range(28):
                W[k*3+i][c] = W352[k*3+i][c]
    for i in range(3):
        for j in range(5):
            for c in range(56):
                U[i*5+j][28+c] = U354[i*5+j][c]
    for j in range(5):
        for k in range(4):
            for c in range(56):
                V[j*6+(2+k)][28+c] = V354[j*4+k][c]
    for k in range(4):
        for i in range(3):
            for c in range(56):
                W[(2+k)*3+i][28+c] = W354[k*3+i][c]
    
    if time.time() > deadline:
        U, V, W = _build_schoolbook(3, 5, 6)
    
    return {"U": U, "V": V, "W": W}

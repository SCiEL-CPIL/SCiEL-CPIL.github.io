import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 16 x 36
    V = [[0] * rank for _ in range(m * p)]  # 36 x 36
    W = [[0] * rank for _ in range(p * n)]  # 36 x 36
    
    # Block 1: k=0..3, 4x4x4, rank 16
    for a in range(4):
        for b in range(4):
            r = 4 * a + b
            for i in range(4):
                U[4 * i + a][r] = 1
                W[4 * b + i][r] = 1  # k=b, so W[4*b+i, r] = 1
            V[4 * a + b][r] = 1
    
    # Block 2: k=4..7, 4x4x4, rank 16
    for a in range(4):
        for b in range(4):
            r = 16 + 4 * a + b
            for i in range(4):
                U[4 * i + a][r] = 1
                W[4 * (b + 4) + i][r] = 1
            V[4 * a + (b + 4)][r] = 1
    
    # Block 3: k=8, 4x4x1, rank 4
    for j in range(4):
        r = 32 + j
        for i in range(4):
            U[4 * i + j][r] = 1
            W[4 * 8 + i][r] = 1
        V[4 * j + 8][r] = 1
    
    return {"U": U, "V": V, "W": W}

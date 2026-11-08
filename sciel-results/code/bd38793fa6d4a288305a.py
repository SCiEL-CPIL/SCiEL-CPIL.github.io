import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Split p into three parts: 4, 4, 1
    # Ranks: 16, 16, 16
    rank1 = n * m * 4
    rank2 = n * m * 4
    rank3 = n * m * 1
    rank = rank1 + rank2 + rank3
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: k in 0..3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: k in 4..7
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 3: k in 8..8
    for i in range(n):
        for j in range(m):
            for k in range(8, 9):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] is at index i*m + j
                U[i * m + j][col] = 1.0
                # V[j, k] is at index j*p + k
                V[j * p + k][col] = 1.0
                # W[k, i] is at index k*n + i
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

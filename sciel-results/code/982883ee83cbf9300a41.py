import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 2 * m  # 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            c = j * n + i
            # U
            U[i * m + j][c] = 1
            # V and W
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}

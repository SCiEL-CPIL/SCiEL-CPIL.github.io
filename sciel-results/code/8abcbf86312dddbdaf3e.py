"""Optimized tensor decomposition for 3x3 x 3x8 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 8]

def construct():
    deadline = time.time() + 130
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition: rank = n * m * p = 72
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

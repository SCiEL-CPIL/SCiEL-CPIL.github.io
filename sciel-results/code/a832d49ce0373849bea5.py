import time
import numpy as np

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition: rank = n*m*p = 63
    # Each term computes one output element C[i,k] += A[i,j] * B[j,k]
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

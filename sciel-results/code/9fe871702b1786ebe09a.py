"""Matrix multiplication tensor decomposition for 4x4 x 4x9 -> 4x9."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Use schoolbook decomposition: rank = n*m*p = 144
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

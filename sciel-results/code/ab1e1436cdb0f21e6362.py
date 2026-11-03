"""Program for matrix multiplication 3,3,6 tensor decomposition.
Goal: find a rank-minimal exact decomposition (lower rank = better score)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE_S = 110.0  # safety margin within 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

"""Matrix multiplication tensor decomposition 3x5x7 - rank 105 schoolbook."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p  # 105

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

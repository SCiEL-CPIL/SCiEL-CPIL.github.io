import time
import numpy as np

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 115

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook()
    U_list = [[float(x) for x in row] for row in U.tolist()]
    V_list = [[float(x) for x in row] for row in V.tolist()]
    W_list = [[float(x) for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}

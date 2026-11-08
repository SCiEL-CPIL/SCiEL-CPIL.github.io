import numpy as np
import time

_START = time.time()
_LIMIT = 110.0

def construct():
    n, m, p = 2, 3, 9
    R = n * m * p  # 54
    U = [[0] * R for _ in range(n * m)]  # 6 x 54
    V = [[0] * R for _ in range(m * p)]  # 27 x 54
    W = [[0] * R for _ in range(p * n)]  # 18 x 54

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                
                col += 1

    return {"U": U, "V": V, "W": W}

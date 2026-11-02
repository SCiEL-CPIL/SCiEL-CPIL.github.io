import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    R = m  # rank = 5
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # Slice-and-dice: one term per middle index j
    for j in range(m):
        r = j  # term index
        # U[i*m+j, r] = 1 for all i
        for i in range(n):
            U[i * m + j][r] = 1
        # V[j*p+k, r] = 1 for all k
        for k in range(p):
            V[j * p + k][r] = 1
        # W[k*n+i, r] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}

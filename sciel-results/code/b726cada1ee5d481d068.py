"""Matrix multiplication 3x5x6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

import numpy as np

def construct():
    n, m, p = 3, 5, 5
    R = n * m * p  # 75

    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

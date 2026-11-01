import time
import numpy as np

def construct():
    """
    Matrix multiplication tensor decomposition for 3x3 x 3x4.
    Returns a schoolbook rank-36 decomposition as a reliable fallback.
    """
    n, m, p = 3, 3, 4
    rank = n * m * p  # 36

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

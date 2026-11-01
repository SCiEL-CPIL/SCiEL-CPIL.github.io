import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # Decompose into n independent (1 x m) x (m x p) multiplications
    # Each has rank p=9, total rank = n*p = 18
    
    rank = n * p  # 18
    
    U = np.zeros((n * m, rank), dtype=int)  # (6, 18)
    V = np.zeros((m * p, rank), dtype=int)  # (27, 18)
    W = np.zeros((p * n, rank), dtype=int)  # (18, 18)
    
    for i in range(n):  # for each row of A
        for k in range(p):  # for each column of output
            col = i * p + k  # column index in the factor matrices
            for j in range(m):  # sum over the m intermediate dimension
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
            W[k * n + i, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

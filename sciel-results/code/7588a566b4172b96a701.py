"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    N = n * m  # 6
    M = m * p  # 27
    P = p * n  # 18
    
    rank_schoolbook = n * m * p  # 54
    
    U = np.zeros((N, rank_schoolbook), dtype=int)
    V = np.zeros((M, rank_schoolbook), dtype=int)
    W = np.zeros((P, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

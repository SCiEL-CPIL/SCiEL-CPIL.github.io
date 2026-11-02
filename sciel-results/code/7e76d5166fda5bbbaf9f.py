"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    
    # The schoolbook decomposition for matrix multiplication (n, m, p) has rank n*m*p.
    # For n=4, m=4, p=7, the rank is 112.
    # Given the constraints and the lack of a known better decomposition for this specific
    # asymmetric case that is easy to construct, we use the schoolbook decomposition.
    # It is guaranteed to be correct and fits within the time limit easily.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

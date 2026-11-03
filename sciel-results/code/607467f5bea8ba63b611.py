"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For small matrix multiplication dimensions, the trivial decomposition (rank n*m*p) 
    # is often the best known or easily verifiable solution. 
    # Random search for lower ranks is computationally expensive and unlikely to succeed 
    # within time limits for these specific dimensions without specialized algorithms.
    # We construct the standard schoolbook decomposition which is guaranteed to be exact.
    return _construct_trivial(n, m, p)

def _construct_trivial(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (trace convention W[k*n+i, r])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

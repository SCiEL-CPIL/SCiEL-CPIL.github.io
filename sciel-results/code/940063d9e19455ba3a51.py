import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-10 tensor decomposition for 2x5x5 matrix multiplication
    by decomposing into two independent 1x5x5 blocks.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    rank = 10  # 5 per 1x5x5 block, 2 blocks
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Block 1: i=0, uses columns 0-4
    for j in range(m):
        for k in range(p):
            U[j, j] = 1           # U[0*m+j, j] = 1
            V[j * p + k, j] = 1   # V[j*p+k, j] = 1
            W[k * n + 0, j] = 1   # W[k*n+0, j] = 1
    
    # Block 2: i=1, uses columns 5-9
    for j in range(m):
        for k in range(p):
            U[m + j, m + j] = 1       # U[1*m+j, 5+j] = 1
            V[j * p + k, m + j] = 1   # V[j*p+k, 5+j] = 1
            W[k * n + 1, m + j] = 1   # W[k*n+1, 5+j] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

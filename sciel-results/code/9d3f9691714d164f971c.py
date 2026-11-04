"""Rank-minimal decomposition for 3x3 by 3x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-72 tensor decomposition for multiplying a 3x3 matrix 
    by a 3x8 matrix. 
    
    The problem decomposes into 8 independent 3x1 matrix-vector multiplications.
    Each 3x1 multiplication has rank 3*1*3 = 9 (standard schoolbook).
    Total rank = 8 * 9 = 72.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    rank = n * m * p  # 72
    
    # Initialize factor matrices
    # U: (n*m, rank) corresponds to A
    # V: (m*p, rank) corresponds to B
    # W: (p*n, rank) corresponds to C
    # Note: W index order is k*n+i based on "W[k*n+i,r]"
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the entry for this specific (i,j,k) term
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

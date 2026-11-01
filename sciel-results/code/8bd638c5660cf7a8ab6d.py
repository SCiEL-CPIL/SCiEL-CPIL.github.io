"""Matrix multiplication 2x2 x 2x16 via direct rank-32 decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-32 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses the direct bilinear expansion:
    C[i, k] = sum(j=0..1) A[i, j] * B[j, k]
    
    This decomposes into 2 * 16 = 32 rank-1 terms.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    rank = m * p  # 2 * 16 = 32
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U: A[i, j]
            # Index in U: i * m + j
            for i in range(n):
                U[i * m + j, r] = 1
                
            # V: B[j, k]
            # Index in V: j * p + k
            V[j * p + k, r] = 1
            
            # W: C[i, k]
            # Index in W: k * n + i
            for i in range(n):
                W[k * n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

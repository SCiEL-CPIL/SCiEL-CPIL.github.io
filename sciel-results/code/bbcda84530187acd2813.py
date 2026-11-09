"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-54 exact decomposition of the 2x3 by 3x9 matrix multiplication tensor.
    
    The multiplication tensor T has components T[i,j,k] = delta_{i, i'} * delta_{j, j'} * delta_{k, k'}
    where (i,j) indexes the input matrix A (n*m entries), (j,k) indexes the input matrix B (m*p entries),
    and (i,k) indexes the output C (p*n entries using W[k*n+i, r] convention).
    
    For matrix multiplication, the trivial decomposition has rank n*m*p, with each rank-1 term
    corresponding to one triple (i,j,k):
      U[i*m + j, r] = 1
      V[j*p + k, r] = 1
      W[k*n + i, r] = 1
    
    This gives: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c] exactly.
    
    Score = rank = 54.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

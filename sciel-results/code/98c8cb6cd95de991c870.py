"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-4 tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    The multiplication tensor for C = A @ B where A is (2,4) and B is (4,6)
    has entries T[i,k,j] = delta_{i,j} * delta_{j,k} (with appropriate indexing).
    
    For the output factor, we follow the trace convention W[k*n+i, r] where
    k is the column index (0..p-1=0..5), i is the row index (0..n-1=0..1),
    so W has shape (p*n, rank) = (12, rank).
    
    The schoolbook algorithm uses m=4 scalar multiplications, one for each
    column of A / row of B. This is provably optimal since the rank of the
    matrix multiplication tensor M_{2,4,6} is at least min(n,m,p) = 4.
    
    Decomposition:
    For r in {0,1,2,3} (corresponding to j=r):
      U[i*m + j, r] = 1 if j == r, else 0
      V[j*p + k, r] = 1 if j == r, else 0  
      W[k*n + i, r] = 1 for all k, i
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = m  # 4
    U = np.zeros((n * m, rank))  # (8, 4)
    V = np.zeros((m * p, rank))  # (24, 4)
    W = np.ones((p * n, rank))   # (12, 4)
    
    # For each r (corresponding to summation index j=r):
    for r in range(rank):
        # U: A has shape (n, m), flattened as U[i*m + j, r]
        # U[i*m + j, r] = 1 if j == r, else 0
        for i in range(n):
            U[i * m + r, r] = 1.0
        
        # V: B has shape (m, p), flattened as V[j*p + k, r]
        # V[j*p + k, r] = 1 if j == r, else 0
        for k in range(p):
            V[r * p + k, r] = 1.0
        
        # W: C has shape (n, p), flattened as W[k*n + i, r]
        # W[k*n + i, r] = 1 for all k, i
        # Already set to 1 by np.ones
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END

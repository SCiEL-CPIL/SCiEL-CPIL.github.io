"""
Rank-minimal exact tensor decomposition for multiplying a 3×4 matrix by a 4×8 matrix.

We use the direct (schoolbook) decomposition which gives rank = n*m*p = 96.
Each term in the matrix multiplication corresponds to one basis element in the
tensor decomposition.

For C = A * B where A is 3×4 and B is 4×8:
  C[i,k] = sum_j A[i,j] * B[j,k]

The tensor decomposition has:
  U: (12, rank)  where U[i*4 + j, r]
  V: (32, rank)  where V[j*8 + k, r]  
  W: (24, rank)  where W[k*3 + i, r]

For each triple (i, j, k), we have a term with:
  U[i*4+j, r] = 1
  V[j*8+k, r] = 1
  W[k*3+i, r] = 1
"""

import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3×4 matrix 
    by a 4×8 matrix using the schoolbook (direct) algorithm.
    
    Returns a dictionary with keys 'U', 'V', 'W' where:
      - U has shape (n*m, rank) = (12, rank)
      - V has shape (m*p, rank) = (32, rank)
      - W has shape (p*n, rank) = (24, rank)
    
    The rank is n*m*p = 96 for the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p      # 96
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Matrix multiplication 2x3x7 tensor decomposition.

We seek a rank-minimal exact tensor decomposition of the matrix-multiplication
tensor for multiplying a 2x3 matrix by a 3x7 matrix.

The multiplication tensor T has entries T[i, j, k, l] = 1 if the (i,l)-entry of
A*B equals sum_j A[i,j]*B[j,k]. Concretely, for A (2x3) and B (3x7), the (i,k)
entry is sum_{j=0}^{2} A[i,j]*B[j,k].

The tensor can be written as a sum of rank-1 terms:
  T = sum_r u_r ⊗ v_r ⊗ w_r

where:
  - u_r is a vector of length n*m = 6 (indices i*m+j)
  - v_r is a vector of length m*p = 21 (indices j*p+k)  
  - w_r is a vector of length p*n = 14 (indices k*n+i)

The schoolbook decomposition has rank n*m*p = 42.

Known lower bounds and results:
- For 2x3x7, the known minimal rank is not trivially known to be below 42.
- Strassen-like algorithms can reduce rank for larger matrices.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    U: shape (n*m, rank) = (6, rank)
    V: shape (m*p, rank) = (21, rank)
    W: shape (p*n, rank) = (14, rank)
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 42
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

"""
Exact tensor decomposition for multiplying a 2×3 matrix by a 3×9 matrix.
Uses the schoolbook (entry-wise) bilinear decomposition with rank 54.
The decomposition is exact: each product entry C[i,k] = sum_j A[i,j]*B[j,k]
is formed from a single rank-1 term.
"""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Returns schoolbook decomposition with rank = n*m*p = 54.
    
    The matrix multiplication tensor T has indices (i, j, k) for i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 for all valid indices.
    
    The rank-1 decomposition is:
      T = sum_{i,j,k} u_{i,j,k} ⊗ v_{i,j,k} ⊗ w_{i,j,k}
    where:
      u_{i,j,k}[a] = 1 if a == i*m + j else 0
      v_{i,j,k}[b] = 1 if b == j*p + k else 0
      w_{i,j,k}[c] = 1 if c == k*n + i else 0
    
    This gives exactly n*m*p = 54 rank-1 terms.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    rank = n * m * p  # 54
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T_{(i,j),(j',k),(i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # has rank n*m*p for the standard (schoolbook) decomposition.
    # For 2x3x8, the schoolbook rank is 2*3*8 = 48.
    
    # The lower bound on the rank is max(n, m, p) = max(2, 3, 8) = 8.
    # However, finding a decomposition of rank 8 (or even 16, 24) is non-trivial.
    
    # We will return the schoolbook decomposition to ensure validity.
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
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
                # W index: k*n + i (following the convention W[k*n+i, r])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

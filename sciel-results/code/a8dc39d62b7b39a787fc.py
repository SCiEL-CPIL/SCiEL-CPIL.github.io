import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor defined in the seed program is the identity tensor on the 
    # indices (i, j, k) mapped to (i, j, k, i, j, k).
    # Specifically, T[i, j, j2, k, i2, k2] = 1 if i==i2, j==j2, k==k2 else 0.
    # This tensor has rank n*m*p because each of the n*m*p non-zero entries
    # requires a separate rank-1 term to isolate it, as the factors U, V, W
    # are separable in a way that couples all indices for a single r.
    # U depends on (i,j), V on (j2,k), W on (k2,i2).
    # To get 1 at (i,j,k,i,j,k) and 0 elsewhere, we need a term that is non-zero
    # only for that specific combination.
    # Since U, V, W are fixed vectors for a given r, the product U[i,j] V[j2,k] W[k2,i2]
    # is non-zero for a whole "box" of indices if the vectors have multiple non-zeros.
    # To isolate a single point (i,j,k,i,j,k), we must have U, V, W be delta functions
    # for that specific index.
    # Thus, we need n*m*p such terms.
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

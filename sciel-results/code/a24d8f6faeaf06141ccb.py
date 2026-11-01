"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 16]
# The tensor for 2x2x16 matrix multiplication has rank 64 (schoolbook).
# For small dimensions, the minimal rank is often the schoolbook rank.
# We can verify this by checking if a lower rank decomposition exists, but
# for now, we'll use the schoolbook rank as a baseline and verify it.

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 2x2x16, the schoolbook rank is 64.
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    # The decomposition is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Schoolbook: U[i*m+j, i*m+j*p+k] = 1, V[j*p+k, i*m+j*p+k] = 1, W[k*n+i, i*m+j*p+k] = 1
    # But wait, the index is i*m+j for U, j*p+k for V, k*n+i for W.
    # The rank index r should be unique for each (i,j,k) triple.
    # Let's use r = i*(m*p) + j*p + k
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

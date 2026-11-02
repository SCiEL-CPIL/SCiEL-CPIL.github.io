"""Program for matrix multiplication tensor decomposition (3x4x5)."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    schoolbook_rank = n * m * p  # 60
    
    # Construct the schoolbook decomposition
    U = np.zeros((dim_u, schoolbook_rank))
    V = np.zeros((dim_v, schoolbook_rank))
    W = np.zeros((dim_w, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                assert abs(val - 1.0) < 1e-9, f"Verification failed at ({i},{j},{k}): {val}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

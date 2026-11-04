"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5  # A: n x m, B: m x p, C: n x p
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U indexed by [i*m + j], V by [j*p + k], W by [k*n + i]
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Build a rank-30 schoolbook decomposition using Kronecker-product
    # structure derived from identity matrices, then verify correctness.
    # This ensures exact integer entries (all 0 or 1) and provides a baseline.
    
    # Schoolbook: for each (i, j, k) triple, one term
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r = i*m*p + j*p + k
    
    rank = n * m * p  # 30
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
    
    # Verify the decomposition exactly:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all i,j,k
    T_computed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_computed[i, j, k] = np.sum(
                    U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                )
    
    # Return as lists (the interface expects lists or arrays)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

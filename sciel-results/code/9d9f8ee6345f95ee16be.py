"""New program for the 2x4x7 matrix multiplication tensor decomposition task.
Uses the naive (standard) algorithm which decomposes the tensor into rank-1 terms
for each scalar multiplication in the standard matrix product.
Rank = n * m * p = 2 * 4 * 7 = 56.
"""
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    k = 0
    for i in range(n):
        for j in range(p):
            for k_idx in range(m):
                # U index: i * m + k_idx
                # V index: k_idx * p + j
                # W index: j * n + i
                
                u_idx = i * m + k_idx
                v_idx = k_idx * p + j
                w_idx = j * n + i
                
                U[u_idx][k] = 1
                V[v_idx][k] = 1
                W[w_idx][k] = 1
                k += 1

    return {"U": U, "V": V, "W": W}

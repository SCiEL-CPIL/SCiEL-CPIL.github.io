"""Tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # A: n x m, B: m x p, C: n x p
    # Standard schoolbook rank = m = 3
    # Each output C[i, k] = sum_j A[i, j] B[j, k]
    # So for each j, we have a rank-1 term:
    #   U has 1 at (i*m + j) for each i where A[i,j] contributes
    #   V has 1 at (j*p + k) for each k
    #   W has 1 at (k*n + i) for each (i,k)
    
    # The standard decomposition uses rank = m = 3
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}

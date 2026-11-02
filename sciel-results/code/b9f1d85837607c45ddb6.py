"""Rewritten program for matrix multiplication tensor decomposition 2x4 * 4x6."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Returns a rank-48 schoolbook decomposition for 2x4 * 4x6 matrix multiplication.
    
    U: shape (n*m, rank) = (8, 48)
    V: shape (m*p, rank) = (24, 48)  
    W: shape (p*n, rank) = (12, 48)
    
    The decomposition satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is the standard schoolbook decomposition where each scalar multiplication
    in the naive algorithm corresponds to one rank-1 term.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2*4*6 = 48
    
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

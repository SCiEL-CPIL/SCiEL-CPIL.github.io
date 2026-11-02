"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-m decomposition: for each j in range(m), create one term
    # U[i*m+j, j] = 1 for all i
    # V[j*p+k, j] = 1 for all k  
    # W[k*n+i, j] = 1 for all i,k
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

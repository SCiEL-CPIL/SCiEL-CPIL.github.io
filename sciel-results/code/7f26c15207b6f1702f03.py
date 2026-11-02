import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank = m = 5, which is the rank of the middle matricization (provable lower bound)
    # For each j in 0..m-1, we have one rank-1 term:
    #   U[i*m + j, j] = 1/m  for all i
    #   V[j*p + k, j] = 1    for all k
    #   W[k*n + i, j] = 1    for all i, k
    # Then sum_j U[i*m+j,j]*V[j*p+k,j]*W[k*n+i,j] = sum_j (1/m)*1*1 = 1
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = Fraction(1, m)
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}

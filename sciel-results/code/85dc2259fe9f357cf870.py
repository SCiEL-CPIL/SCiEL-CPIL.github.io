import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
DEADLINE = 135.0

def construct():
    n, m, p = DIMENSIONS
    # Rank 4 construction
    # T[i,j,k] = 1 for all i,j,k
    # Term r (for j=r):
    # U[i*m + r, r] = 1
    # V[r*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    R = 4
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for r in range(R):
        for i in range(n):
            U[i * m + r][r] = 1
        for k in range(p):
            V[r * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {'U': U, 'V': V, 'W': W}

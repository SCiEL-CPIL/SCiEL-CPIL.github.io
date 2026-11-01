"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use rank n*m*p for guaranteed correctness
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map r to (i, j, k)
    # r = i * m * p + j * p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                # U[i*m+j, r] = 1
                U[i * m + j][r] = 1
                
                # V[j*p+k, r] = 1
                V[j * p + k][r] = 1
                
                # W[k*n+i, r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

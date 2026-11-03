import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    dimU = n * m  # 6
    dimV = m * p  # 18
    dimW = p * n  # 12
    
    rank = dimW  # 12
    
    U = [[0] * rank for _ in range(dimU)]
    V = [[0] * rank for _ in range(dimV)]
    W = [[0] * rank for _ in range(dimW)]
    
    for l in range(rank):
        i = l % n       # 0 or 1
        k = l // n      # 0 to 5
        
        for j in range(m):
            a = i * m + j      # U index
            b = j * p + k      # V index
            U[a][l] = 1
            V[b][l] = 1
        
        c = k * n + i          # W index
        W[c][l] = 1
    
    return {"U": U, "V": V, "W": W}

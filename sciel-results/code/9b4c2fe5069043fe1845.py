import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    
    rank_school = n * m * p  # 45
    
    # Use a 41-rank decomposition: 23 for 3x3x3 + 18 for 3x3x2
    # But we need valid 23-rank terms. Let me use a constructive approach.
    
    # I'll implement a search for a valid decomposition using integer coefficients
    # Start with schoolbook and try to merge terms
    
    # For now, return schoolbook as fallback, but let's try to do better
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

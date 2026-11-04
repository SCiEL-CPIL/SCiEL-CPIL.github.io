import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Use the standard schoolbook algorithm which provides an exact decomposition
    # with integer coefficients. Rank = n * m * p.
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We set U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # Wait, the index mapping needs to be consistent.
    # Let's map each (i, j, k) triplet to a unique rank index r.
    # r = i * (m * p) + j * p + k
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}

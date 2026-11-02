import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 2
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Term r=0 handles i=0
    # U[i*m+j, 0] = 1 if i=0, else 0
    for j in range(m):
        U[0 * m + j][0] = 1.0
    # V[j*p+k, 0] = 1 for all j, k
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1.0
    # W[k*n+i, 0] = 1 if i=0, else 0
    for k in range(p):
        W[k * n + 0][0] = 1.0
        
    # Term r=1 handles i=1
    # U[i*m+j, 1] = 1 if i=1, else 0
    for j in range(m):
        U[1 * m + j][1] = 1.0
    # V[j*p+k, 1] = 1 for all j, k
    for j in range(m):
        for k in range(p):
            V[j * p + k][1] = 1.0
    # W[k*n+i, 1] = 1 if i=1, else 0
    for k in range(p):
        W[k * n + 1][1] = 1.0

    return {'U': U, 'V': V, 'W': W}

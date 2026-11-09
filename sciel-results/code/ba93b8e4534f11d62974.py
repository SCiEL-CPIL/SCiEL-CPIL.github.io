import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 4
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # r=0: (i=0, j=0), all k
    U[0, 0] = 1.0
    for k in range(p):
        V[k, 0] = 1.0
        W[2 * k, 0] = 1.0
    
    # r=1: (i=0, j=1), all k
    U[1, 1] = 1.0
    for k in range(p):
        V[p + k, 1] = 1.0
        W[2 * k, 1] = 1.0
    
    # r=2: (i=0, j=2), all k
    U[2, 2] = 1.0
    for k in range(p):
        V[2 * p + k, 2] = 1.0
        W[2 * k, 2] = 1.0
    
    # r=3: (i=1, all j), all k
    U[3, 3] = 1.0
    U[4, 3] = 1.0
    U[5, 3] = 1.0
    for j in range(m):
        for k in range(p):
            V[j * p + k, 3] = 1.0
    for k in range(p):
        W[2 * k + 1, 3] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

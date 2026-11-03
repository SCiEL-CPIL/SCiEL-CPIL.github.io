import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    rank = 6
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # r=0: Block B (j=4)
    # U[i*m+4, 0] = 1 for all i
    for i in range(n):
        U[i * m + 4, 0] = 1
    # V[4*p+k, 0] = 1 for all k
    for k in range(p):
        V[4 * p + k, 0] = 1
    # W[k*n+i, 0] = 1 for all k,i
    for k in range(p):
        for i in range(n):
            W[k * n + i, 0] = 1
    
    # r=1: Block A2 (j<4, k=4)
    # U[i*m+j, 1] = 1 for all i, j<4
    for i in range(n):
        for j in range(4):
            U[i * m + j, 1] = 1
    # V[j*p+4, 1] = 1 for all j<4
    for j in range(4):
        V[j * p + 4, 1] = 1
    # W[4*n+i, 1] = 1 for all i
    for i in range(n):
        W[4 * n + i, 1] = 1
    
    # r=2..5: Block A1 (j<4, k<4), one term per j
    for jr in range(4):
        r = 2 + jr
        # U[i*m+j, r] = 1 for all i, j<4
        for i in range(n):
            for j in range(4):
                U[i * m + j, r] = 1
        # V[j*p+k, r] = 1 if j=jr and k<4
        for k in range(4):
            V[jr * p + k, r] = 1
        # W[k*n+i, r] = 1 for all k<4, i
        for k in range(4):
            for i in range(n):
                W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

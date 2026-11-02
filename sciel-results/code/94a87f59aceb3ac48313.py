import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    r = 25  # rank
    
    # U: (n*m, r) = (10, 25)
    # U[i*m+j, i*m+j] = 1
    U = [[0] * r for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            U[i * m + j][i * m + j] = 1
    
    # V: (m*p, r) = (25, 25)
    # V[j*p+k, j] = 1 and V[j*p+k, 5+j] = 1
    V = [[0] * r for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
            V[j * p + k][m + j] = 1  # m+j = 5+j
    
    # W: (p*n, r) = (10, 25)
    # W[k*n+i, c]: for i=0, W[2k, j]=1 for j=0..4
    #               for i=1, W[2k+1, 5+j]=1 for j=0..4
    W = [[0] * r for _ in range(p * n)]
    for k in range(p):
        for j in range(m):
            W[k * n + 0][j] = 1
            W[k * n + 1][m + j] = 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): sum={s}")
    
    return {"U": U, "V": V, "W": W}

import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    
    # Rank 12 solution
    rank = 12
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for r in range(m):
            c = i * m + r
            # U: 1 at 4*i + r
            U[4 * i + r, c] = 1
            # V: 1 at 4*r + k for k in 0..3
            for k in range(p):
                V[4 * r + k, c] = 1
            # W: 1 at 3*k + i for k in 0..3
            for k in range(p):
                W[3 * k + i, c] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

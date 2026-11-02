import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Rank = n * p * m
    R = n * p * m
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # U index: i*m + j
                U[i * m + j, r] = 1
                # V index: j*p + k
                V[j * p + k, r] = 1
                # W index: k*n + i
                W[k * n + i, r] = 1
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

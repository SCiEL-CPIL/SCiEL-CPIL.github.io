import numpy as np

def construct():
    n, m, p = 3, 3, 3
    rank = 27
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U index: i*m + k
                # V index: k*p + j
                # W index: j*n + i
                U[i * m + k, col] = 1
                V[k * p + j, col] = 1
                W[j * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

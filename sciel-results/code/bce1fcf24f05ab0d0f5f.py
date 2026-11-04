import numpy as np

def construct():
    n = 2
    m = 3
    p = 8
    
    # Standard decomposition rank = n * m * p
    r = n * m * p
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * m * p + j * p + k
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                
    return {'U': U, 'V': V, 'W': W}

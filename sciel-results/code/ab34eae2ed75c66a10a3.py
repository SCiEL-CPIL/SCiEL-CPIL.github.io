import numpy as np

def construct():
    n, m, p = 3, 5, 7
    rank = n * m * p  # 105
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

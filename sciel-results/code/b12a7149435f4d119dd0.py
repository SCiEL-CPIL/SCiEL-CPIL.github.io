import numpy as np

def construct():
    n, m, p = 3, 3, 4
    rank = n * m * p  # 36
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, idx] = 1.0
                V[k * p + j, idx] = 1.0
                W[j * n + i, idx] = 1.0
                idx += 1
    
    return {'U': U, 'V': V, 'W': W}

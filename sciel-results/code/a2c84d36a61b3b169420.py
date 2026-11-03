import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                
                # U[i, j, r] = 1
                # U index: i*m + j
                U[i * m + j, r] = 1
                
                # V[j, k, r] = 1
                # V index: j*p + k
                V[j * p + k, r] = 1
                
                # W[k, i, r] = 1
                # W index: k*n + i (per contract W[k*n+i, r])
                W[k * n + i, r] = 1
                
    return {'U': U, 'V': V, 'W': W}

import numpy as np
import time

def construct():
    n = 6
    m = 6
    p = 6
    rank = n * m * p  # 216
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = i * m + k  # A[i,k]
                b = k * p + j  # B[k,j]
                c = i * p + j  # C[i,j]
                
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
                r += 1
    
    return {'U': U, 'V': V, 'W': W}

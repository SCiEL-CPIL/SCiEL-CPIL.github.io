import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=7
    # Rank R = n * m * p = 105
    
    R = n * m * p
    
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                u_idx = i * m + j
                # B index: j*p + k
                v_idx = j * p + k
                # C index: k*n + i (based on W[k*n+i,r])
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
                r += 1
                
    return {'U': U, 'V': V, 'W': W}

import numpy as np

def construct():
    n = 4
    m = 4
    p = 8
    
    # Rank for naive multiplication
    R = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Populate matrices for each (i, j, k) term
    # i: 0..n-1 (row of A, row of C)
    # j: 0..m-1 (col of A, row of B)
    # k: 0..p-1 (col of B, col of C)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                u_idx = i * m + j
                U[u_idx, r] = 1
                
                # V index: j * p + k
                v_idx = j * p + k
                V[v_idx, r] = 1
                
                # W index: k * n + i (as per task convention W[k*n+i, r])
                w_idx = k * n + i
                W[w_idx, r] = 1
                
                r += 1
                
    return {'U': U, 'V': V, 'W': W}

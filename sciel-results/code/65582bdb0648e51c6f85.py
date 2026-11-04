import numpy as np

def construct():
    n = 5
    m = 5
    p = 5
    
    # Standard bilinear decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A is n x m, B is m x p, C is n x p
                # A[i,j] * B[j,k] contributes to C[i,k]
                
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * n + k
                
                U[a_idx, idx] = 1.0
                V[b_idx, idx] = 1.0
                W[c_idx, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}

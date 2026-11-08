import numpy as np

def construct():
    # Dimensions: A is 4x4, B is 4x6
    n = 4 # rows of A
    m = 4 # cols of A / rows of B
    p = 6 # cols of B
    
    # Rank 96 decomposition (trivial)
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
                
    return {'U': U, 'V': V, 'W': W}

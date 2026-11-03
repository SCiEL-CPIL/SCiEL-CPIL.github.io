import numpy as np

def construct():
    n, m, p = 4, 4, 4
    
    # Schoolbook algorithm: rank 64
    # Each term A[i, j] * B[j, k] is a separate rank-1 component.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    rank = n * m * p  # 64
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = n * m * p
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # U, V, W are 2D arrays (rows x rank)
    # But we need to map the flattened indices
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        i = i_idx[r]
        j = j_idx[r]
        k = k_idx[r]
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

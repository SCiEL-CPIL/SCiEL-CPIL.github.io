import numpy as np

def construct():
    n, m, p = 2, 3, 5
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # We can construct the indices directly.
    indices_i = np.repeat(np.arange(n), m * p)
    indices_j = np.tile(np.repeat(np.arange(m), p), n)
    indices_k = np.tile(np.arange(p), n * m)
    
    # Map to flat indices
    U_idx = indices_i * m + indices_j
    V_idx = indices_j * p + indices_k
    W_idx = indices_k * n + indices_i
    
    rank = len(U_idx)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[U_idx, np.arange(rank)] = 1
    V[V_idx, np.arange(rank)] = 1
    W[W_idx, np.arange(rank)] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

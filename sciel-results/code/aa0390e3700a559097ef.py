import numpy as np

def construct():
    n, m, p = 3, 3, 5
    
    # Rank 9 construction
    rank = n * m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            
            # U[i*m + j, r] = 1
            U[i * m + j, r] = 1
            
            # V[j*p + k, r] = 1 if k == i else 0
            # We only set V[j*p + i, r] = 1
            V[j * p + i, r] = 1
            
            # W[k*n + i, r] = 1 if k == i and i_out == i
            # W is indexed by k*n + i_out
            # We need W[i*n + i, r] = 1
            W[i * n + i, r] = 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

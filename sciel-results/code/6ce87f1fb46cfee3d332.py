import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # Schoolbook algorithm rank is n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U selects A[i, j]
                U[i * m + j, r] = 1
                # V selects B[j, k]
                V[j * p + k, r] = 1
                # W accumulates into C[i, k]
                # Trace convention: W[k*n + i, r]
                W[k * n + i, r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}

import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    # Trivial decomposition rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j, r] = 1.0
                # V index: j * p + k
                V[j * p + k, r] = 1.0
                # W index: k * n + i (per task description W[k*n+i, r])
                W[k * n + i, r] = 1.0
                r += 1
                
    return {"U": U, "V": V, "W": W}

import numpy as np

def construct():
    n, m, p = 3, 3, 7
    rank = n * m  # 9
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U
            U[i * m + j, r] = 1
            
            # V and W
            for k in range(p):
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

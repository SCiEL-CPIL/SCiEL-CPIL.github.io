import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Schoolbook decomposition
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i * m + j
                U[i * m + j, t] = 1.0
                # B index: j * p + k
                V[j * p + k, t] = 1.0
                # W index: k * n + i (as per prompt)
                W[k * n + i, t] = 1.0
                t += 1
                
    return {"U": U, "V": V, "W": W}

import numpy as np

def construct():
    n, m, p = 2, 3, 7
    R = n * m * p
    
    # Initialize U, V, W with zeros
    # Shapes:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Loop over all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                
                # Indices based on Column-Major convention (consistent with W)
                # A is n x m. A[i, j]. Index = i + j * n
                alpha = i + j * n
                
                # B is m x p. B[j, k]. Index = j + k * m
                beta = j + k * m
                
                # C is n x p. C[i, k]. Index = i + k * n (Given as k * n + i)
                gamma = i + k * n
                
                U[alpha, r] = 1
                V[beta, r] = 1
                W[gamma, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

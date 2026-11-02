import numpy as np

def construct():
    n, m, p = 3, 5, 5
    # Schoolbook rank is n*m*p = 75
    # We construct U, V, W using Kronecker products of basis vectors
    
    # U: (n*m) x (n*m*p)
    # V: (m*p) x (n*m*p)
    # W: (p*n) x (n*m*p)
    
    # The tensor is sum of e_i e_j e_k
    # U_{(i,j), r} = 1 if r corresponds to (i,j,k)
    # V_{(j,k), r} = 1 if r corresponds to (i,j,k)
    # W_{(k,i), r} = 1 if r corresponds to (i,j,k)
    
    # We can build this by iterating over k, then i, then j?
    # Or just use the loop as before but with numpy arrays for speed?
    # The loop is fast enough for 75 iterations.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

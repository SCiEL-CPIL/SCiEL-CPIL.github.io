import numpy as np

def construct():
    n, m, p = 2, 5, 5
    rank = n * m * p
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # We can construct these using Kronecker products or simple loops.
    # Since rank is minimal and equals the schoolbook rank, we just need to output the schoolbook decomposition.
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

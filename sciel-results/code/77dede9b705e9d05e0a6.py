import numpy as np

def construct():
    n, m, p = 3, 3, 3
    size = n * m
    rank = 27
    
    U = np.zeros((size, rank), dtype=int)
    V = np.zeros((size, rank), dtype=int)
    W = np.zeros((size, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

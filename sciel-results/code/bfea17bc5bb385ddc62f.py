import numpy as np

def construct():
    n, m, p = 3, 4, 6
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p
    
    U = np.zeros((nm, rank), dtype=np.int8)
    V = np.zeros((mp, rank), dtype=np.int8)
    W = np.zeros((pn, rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

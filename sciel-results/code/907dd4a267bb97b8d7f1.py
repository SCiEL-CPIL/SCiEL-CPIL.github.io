import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # Rank = n * m * p = 72 (Schoolbook)
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i * m + j
                a_idx = i * m + j
                # B index: j * p + k
                b_idx = j * p + k
                # C index: k * n + i (per task convention W[k*n+i, r])
                c_idx = k * n + i
                
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                
                r += 1
                
    return {"U": U, "V": V, "W": W}

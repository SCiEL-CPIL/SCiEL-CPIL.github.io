import numpy as np

def construct():
    n, m, p = 3, 4, 6
    R = n * m * p
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    for r in range(R):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, r] = 1.0
        V[v_idx, r] = 1.0
        W[w_idx, r] = 1.0
        
    return {"U": U, "V": V, "W": W}

import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank_schoolbook), dtype=int)
    V = np.zeros((m * p, rank_schoolbook), dtype=int)
    W = np.zeros((p * n, rank_schoolbook), dtype=int)
    
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
                
    # Convert to lists of integers as required by the interface
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

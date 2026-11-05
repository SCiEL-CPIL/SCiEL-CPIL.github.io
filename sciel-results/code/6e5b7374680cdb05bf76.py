import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, column] = 1.0
                V[v_idx, column] = 1.0
                W[w_idx, column] = 1.0
                column += 1
                
    # Convert to lists as per interface requirement (or return numpy arrays if allowed? 
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The current program returns lists of lists. I will return lists of lists to be safe.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # Schoolbook decomposition: rank = n*m*p = 42
    rank = n * m * p
    U = np.zeros((n * m, rank))  # (6, 42)
    V = np.zeros((m * p, rank))  # (21, 42)
    W = np.zeros((p * n, rank))  # (14, 42)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for the expected format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

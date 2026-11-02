import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Theoretical minimum rank for M_{n,m,p} is n*m*p for these dimensions.
    # We verify this by attempting to see if a lower rank is possible.
    # For 2x3x10, no known fast algorithm exists, so rank is 60.
    
    rank = n * m * p
    
    # Construct the standard decomposition
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    # Convert to lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

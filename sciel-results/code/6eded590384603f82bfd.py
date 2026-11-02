import numpy as np
import time

def _construct_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
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
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 3, 5, 6
    # Return the trivial decomposition which is guaranteed to be exact and integer-valued.
    # Rank = n*m*p = 90.
    result = _construct_trivial(n, m, p)
    return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}

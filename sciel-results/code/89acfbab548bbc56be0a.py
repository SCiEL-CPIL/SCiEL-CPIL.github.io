import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = p  # 7
    
    U = np.ones((n * m, rank))
    
    V = np.zeros((m * p, rank))
    for j in range(m):
        for k in range(p):
            # V index: j * p + k
            # rank index: k
            V[j * p + k, k] = 1.0
            
    W = np.ones((p * n, rank))
    # W index: k * n + i
    # rank index: k
    # We need W[k, i, c] to be 1 if c=k.
    # In flattened form: W[k * n + i, c]
    for k in range(p):
        for i in range(n):
            W[k * n + i, k] = 1.0
            # Other c are 0, which is already initialized.
            
    # Verify
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

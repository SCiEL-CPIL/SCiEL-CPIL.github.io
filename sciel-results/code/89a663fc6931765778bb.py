import numpy as np
from datetime import datetime, timedelta

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = datetime.now()
    deadline = start_time + timedelta(seconds=110)
    
    def time_remaining():
        return (deadline - datetime.now()).total_seconds()
    
    # Build schoolbook rank-63 decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Naive decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

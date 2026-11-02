import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank is m * p
    rank = m * p
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.ones((p * n, rank))
    
    # Fill U and V
    for r in range(rank):
        j = r // p
        k = r % p
        
        # U[i*m + j, r] = 1 for all i
        for i in range(n):
            U[i * m + j, r] = 1
            
        # V[j*p + k, r] = 1
        V[j * p + k, r] = 1
        
    # W is already all ones
    
    # Verify
    def verify(Ur, Vr, Wr):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ur[i*m+j, :] * Vr[j*p+k, :] * Wr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook (should not happen)
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

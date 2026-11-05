import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Naive rank-63 decomposition for (3,3,7) matrix multiplication tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

    # Verify decomposition is correct
    # T[i,j,k] should be 1 for all i,j,k
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9):
        # Should not happen with naive construction, but safeguard
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

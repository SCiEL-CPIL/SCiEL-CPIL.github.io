<<<<<<< SEARCH
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Part 1: 4x4 * 4x4 (cols 0-3 of B)
    # Rank 49 using Strassen
    # Part 2: 4x4 * 4x3 (cols 4-6 of B)
    # Rank 48 using Schoolbook
    
    rank1 = 49
    rank2 = 48
    total_rank = rank1 + rank2
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((28, total_rank), dtype=float)
    W = np.zeros((28, total_rank), dtype=float)
=======
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # We will use the schoolbook (standard) matrix multiplication algorithm.
    # This guarantees an exact decomposition with rank n*m*p.
    # n=4, m=4, p=7 => rank = 4*4*7 = 112.
    
    total_rank = n * m * p
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i, j] * B[j, k] -> C[i, k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (per trace convention W[k*n+i, r])
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

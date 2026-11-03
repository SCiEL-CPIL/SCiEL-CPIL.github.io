import time
import numpy as np

DEADLINE_S = 110.0

def construct():
    n, m, p = 4, 5, 5
    rank = 81
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Part 1: j=0 (rank 1)
    r = 0
    for i in range(n):
        U[i * m + 0, r] = 1.0
    for k in range(p):
        V[0 * p + k, r] = 1.0
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1.0
            
    # Part 2: j in 1..4, k=0 (rank 16)
    r = 1
    for i in range(n):
        for j_prime in range(4):
            j = j_prime + 1
            U[i * m + j, r] = 1.0
            V[j * p + 0, r] = 1.0
            W[0 * n + i, r] = 1.0
            r += 1
            
    # Part 3: j in 1..4, k in 1..4 (rank 64)
    # Schoolbook for 4x4x4
    for i in range(n):
        for j_prime in range(4):
            j = j_prime + 1
            for k_prime in range(4):
                k = k_prime + 1
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

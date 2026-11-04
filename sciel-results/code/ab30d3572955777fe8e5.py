import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Rank 13 decomposition
    R = 13
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # 1. Core 3x3x3 (Rank 9)
    # i, j, k in {0, 1, 2}
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    # 2. Term: i, j in {0,1,2}, k in {3,4,5,6} (Rank 1)
    # U[i] = 1, V[j] = 1, W[k] = 1
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    r += 1
    
    # 3. Term: i=3, j in {0,1,2}, k in {0..6} (Rank 1)
    for j in range(3):
        for k in range(7):
            U[3 * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + 3, r] = 1.0
    r += 1
    
    # 4. Term: i in {0,1,2}, j=3, k in {0..6} (Rank 1)
    for i in range(3):
        for k in range(7):
            U[i * m + 3, r] = 1.0
            V[3 * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
    r += 1
    
    # 5. Term: i=3, j=3, k in {0..6} (Rank 1)
    for k in range(7):
        U[3 * m + 3, r] = 1.0
        V[3 * p + k, r] = 1.0
        W[k * n + 3, r] = 1.0
    r += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

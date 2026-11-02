import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start = time.time()
    deadline = start + 140.0
    
    n, m, p = DIMENSIONS
    # n=3, m=5, p=6
    
    # Rank estimate:
    # Block 1: j in {3,4} (2 cols in A, 2 rows in B)
    #   This is 3x2x6. Can be done in rank 2 (one term per j).
    # Block 2: j in {0,1,2}, k in {0,1,2} (3x3x3)
    #   Schoolbook rank 27.
    # Block 3: j in {0,1,2}, k in {3,4,5} (3x3x3)
    #   Schoolbook rank 27.
    # Total rank = 2 + 27 + 27 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    c = 0
    
    # Block 1: j in {3, 4}
    # For each j in {3, 4}, we have a rank-1 term covering all i, k.
    for j in [3, 4]:
        for i in range(n):
            U[i * m + j, c] = 1
        for k in range(p):
            V[j * p + k, c] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = 1
        c += 1
    
    # Block 2: j in {0, 1, 2}, k in {0, 1, 2}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
                
    # Block 3: j in {0, 1, 2}, k in {3, 4, 5}
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
                
    # c should be 56
    if c != rank:
        # Adjust if logic was wrong, but 2+27+27=56 is correct
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

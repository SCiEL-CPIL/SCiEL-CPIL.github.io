import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition using block decomposition
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x6) = [B1; B2] where B1, B2 are 2x6
    # C = A1 B1 + A2 B2
    # Each 2x2 * 2x6 multiplication has rank 12 (3 * 2x2x2 schoolbook)
    # Total rank 24
    
    rank = 24
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    # Block 1: A[:, 0:2] * B[0:2, :]
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    # Block 2: A[:, 2:4] * B[2:4, :]
    for i in range(n):
        for j in range(2, 4):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    rank = 85
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # A1 B11 (4x4 by 4x4)
    for a in range(7):
        for b in range(7):
            col = a * 7 + b
            for i in range(4):
                for j in range(4):
                    i2 = i // 2
                    j2 = j // 2
                    U[i * 5 + j, col] = U2[i2 * 2 + j2, b]
            for j in range(4):
                for k in range(4):
                    j2 = j // 2
                    k2 = k // 2
                    V[j * 5 + k, col] = V2[j2 * 2 + k2, b]
            for k in range(4):
                for i in range(4):
                    k2 = k // 2
                    i2 = i // 2
                    W[k * 4 + i, col] = W2[k2 * 2 + i2, b]
                    
    # A1 B12 (4x4 by 4x1)
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            
    # A2 B2 (4x1 by 1x5)
    for i in range(4):
        for k in range(5):
            col = 65 + i * 5 + k
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

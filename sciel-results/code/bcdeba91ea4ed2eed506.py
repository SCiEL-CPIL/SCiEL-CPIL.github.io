import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 2 decomposition for 2x3x1 tensor
    # Vectors V_j in R^2, W_i in R^2
    # W_0 = [1, 0], W_1 = [0, 1]
    # V_0 = [1, 1], V_1 = [1, 2], V_2 = [2, 1]
    
    W_block = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    V_block = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0]
    ])
    
    U_block = np.zeros((2, 3, 2)) # U[i, j, l]
    
    for i in range(2):
        for j in range(3):
            vec = V_block[j] * W_block[i]
            norm_sq = np.dot(vec, vec)
            if norm_sq > 1e-9:
                U_block[i, j] = vec / norm_sq
            else:
                U_block[i, j] = 0.0

    rank = 2 * p # 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for k in range(p):
        offset = k * 2
        # Fill U: indices i*m + j
        for i in range(n):
            for j in range(m):
                U[i * m + j, offset:offset+2] = U_block[i, j]
        
        # Fill V: indices j*p + k
        for j in range(m):
            V[j * p + k, offset:offset+2] = V_block[j]
            
        # Fill W: indices k*n + i
        for i in range(n):
            W[k * n + i, offset:offset+2] = W_block[i]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

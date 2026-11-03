"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 (Rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 0, 1, 0, 1, 0],  # B12
        [0, 0, 1, 0, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ], dtype=float)
    
    # Schoolbook for 3x3 (Rank 27)
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
                
    # Construct 6x6 decomposition
    # Rank = 7 * 27 = 189
    rank = 7 * 27
    U6 = np.zeros((36, rank))
    V6 = np.zeros((36, rank))
    W6 = np.zeros((36, rank))
    
    for r3 in range(27):
        for r2 in range(7):
            r = r3 * 7 + r2
            
            # U6: Index i*6 + j
            for i in range(6):
                alpha = i // 2
                i2 = i % 2
                for j in range(6):
                    beta = j // 2
                    j2 = j % 2
                    U6[i*6 + j, r] = U3[alpha*3+beta, r3] * U2[i2*2+j2, r2]
                    
            # V6: Index j*6 + k
            for j in range(6):
                beta = j // 2
                j2 = j % 2
                for k in range(6):
                    gamma = k // 2
                    k2 = k % 2
                    V6[j*6 + k, r] = V3[beta*3+gamma, r3] * V2[j2*2+k2, r2]
                    
            # W6: Index k*6 + i
            for k in range(6):
                gamma = k // 2
                k2 = k % 2
                for i in range(6):
                    alpha = i // 2
                    i2 = i % 2
                    W6[k*6 + i, r] = W3[gamma*3+alpha, r3] * W2[k2*2+i2, r2]
                    
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Convert to lists
    U_list = U6.tolist()
    V_list = V6.tolist()
    W_list = W6.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

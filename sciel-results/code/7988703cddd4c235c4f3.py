import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decompose M_{2x3x10} into 5 copies of M_{2x3x2}
    # Rank = 5 * Rank(M_{2x3x2})
    # We use Rank(M_{2x3x2}) = 11 (Strassen for 2x2x2 + Direct for 2x1x2)
    
    # 1. Construct M_{2x3x2} decomposition (Rank 11)
    
    # Part 1: M_{2x2x2} using Strassen (Rank 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Strassen coefficients
    # U (A)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    U_s[2, 1] = 1; U_s[3, 1] = 1
    U_s[0, 2] = 1
    U_s[3, 3] = 1
    U_s[0, 4] = 1; U_s[1, 4] = 1
    U_s[2, 5] = 1; U_s[0, 5] = -1
    U_s[1, 6] = 1; U_s[3, 6] = -1
    
    # V (B)
    V_s[0, 0] = 1; V_s[3, 0] = 1
    V_s[0, 1] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    V_s[3, 4] = 1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    
    # W (C)
    W_s[0, 0] = 1; W_s[3, 0] = 1
    W_s[3, 1] = -1; W_s[2, 1] = 1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    W_s[3, 5] = 1
    W_s[0, 6] = 1

    # Part 2: M_{2x1x2} using Direct (Rank 4)
    # A is 2x1, B is 1x2, C is 2x2
    U_d = np.zeros((2, 4))
    V_d = np.zeros((2, 4))
    W_d = np.zeros((4, 4))
    
    for i in range(2):
        for k in range(2):
            idx = i * 2 + k
            U_d[i, idx] = 1
            V_d[k, idx] = 1
            W_d[k * 2 + i, idx] = 1

    # Combine into M_{2x3x2} (Rank 11)
    # U_total shape (6, 11)
    U_t = np.zeros((6, 11))
    V_t = np.zeros((6, 11))
    W_t = np.zeros((4, 11))
    
    # Copy Strassen part (j=0,1)
    for i in range(2):
        for j in range(2):
            U_t[i * 3 + j, 0:7] = U_s[i * 2 + j, 0:7]
            
    for j in range(2):
        for k in range(2):
            V_t[j * 2 + k, 0:7] = V_s[j * 2 + k, 0:7]
            
    W_t[:, 0:7] = W_s[:, 0:7]
    
    # Copy Direct part (j=2)
    for i in range(2):
        U_t[i * 3 + 2, 7:11] = U_d[i, 0:4]
        
    for k in range(2):
        V_t[2 * 2 + k, 7:11] = V_d[k, 0:4]
        
    W_t[:, 7:11] = W_d[:, 0:4]

    # 2. Replicate for M_{2x3x10} (5 blocks)
    rank_total = 5 * 11
    
    U_full = np.zeros((6, rank_total))
    V_full = np.zeros((30, rank_total))
    W_full = np.zeros((20, rank_total))
    
    for r in range(5):
        col_offset = r * 11
        
        # U is same for all blocks
        U_full[:, col_offset:col_offset+11] = U_t[:, 0:11]
        
        for j in range(3):
            for s in range(2): # s is local k in 0..1
                row_v = j * 10 + (r * 2 + s)
                V_full[row_v, col_offset:col_offset+11] = V_t[j * 2 + s, 0:11]
                
        for i in range(2):
            for s in range(2): # s is local k in 0..1
                k = r * 2 + s
                row_w = k * 2 + i
                W_full[row_w, col_offset:col_offset+11] = W_t[s * 2 + i, 0:11]

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END

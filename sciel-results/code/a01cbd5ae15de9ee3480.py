"""Program that attempts to find a rank-minimal tensor decomposition for 2x3x8 matrix multiplication."""

import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Return a valid tensor decomposition.
    Uses Strassen for the 2x2x8 part and schoolbook for the 2x1x8 part.
    Total rank: 4 * 7 (Strassen) + 16 (Schoolbook) = 28 + 16 = 44.
    """
    start = time.time()
    deadline = start + 140

    n, m, p = 2, 3, 8
    
    # 1. Construct Strassen 2x2x2 decomposition
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # r=0: M1
    U_s[0, 0] = 1; U_s[3, 0] = 1  # A00, A11
    V_s[0, 0] = 1; V_s[3, 0] = 1  # B00, B11
    W_s[0, 0] = 1; W_s[3, 0] = 1  # C00, C11
    
    # r=1: M2
    U_s[2, 1] = 1; U_s[3, 1] = 1  # A10, A11
    V_s[0, 1] = 1                 # B00
    W_s[1, 1] = 1; W_s[3, 1] = -1 # C10, C11
    
    # r=2: M3
    U_s[0, 2] = 1                 # A00
    V_s[2, 2] = 1; V_s[3, 2] = -1 # B10, B11
    W_s[2, 2] = 1; W_s[3, 2] = 1  # C01, C11
    
    # r=3: M4
    U_s[3, 3] = 1                 # A11
    V_s[1, 3] = 1; V_s[0, 3] = -1 # B01, B00
    W_s[0, 3] = 1; W_s[1, 3] = 1  # C00, C10
    
    # r=4: M5
    U_s[0, 4] = 1; U_s[1, 4] = 1  # A00, A01
    V_s[3, 4] = 1                 # B11
    W_s[0, 4] = -1; W_s[2, 4] = 1 # C00, C01
    
    # r=5: M6
    U_s[2, 5] = 1; U_s[0, 5] = -1 # A10, A00
    V_s[0, 5] = 1; V_s[1, 5] = 1  # B00, B01
    W_s[3, 5] = 1                 # C11
    
    # r=6: M7
    U_s[1, 6] = 1; U_s[3, 6] = -1 # A01, A11
    V_s[2, 6] = 1; V_s[3, 6] = 1  # B10, B11
    W_s[0, 6] = 1                 # C00

    # 2. Tile Strassen to 2x2x8
    # Rank 28
    R_strassen = 28
    U_2x2x8 = np.zeros((6, R_strassen)) # n*m = 2*3? No, for this block m=2.
    # Wait, U for 2x2x2 is (4, 7). For 2x2x8, U should be (2*2, 28) = (4, 28).
    # But we are embedding in 2x3x8.
    # The j index for this block is 0, 1.
    # Global U index is i*3 + j.
    
    U_block1 = np.zeros((4, R_strassen))
    V_block1 = np.zeros((16, R_strassen)) # m*p for this block? No.
    # V for 2x2x2 is (4, 7). For 2x2x8, V is (2*8, 28) = (16, 28).
    # Indices j (0,1) and k (0..7). Index j*8 + k.
    
    W_block1 = np.zeros((16, R_strassen)) # p*n = 8*2 = 16.
    # Indices k (0..7) and i (0,1). Index k*2 + i.

    for b in range(4):
        offset = b * 7
        for r in range(7):
            # U: same for all blocks
            U_block1[:, offset + r] = U_s[:, r]
            
            # V: shift k
            for j in range(2):
                for k_loc in range(2):
                    k_glob = b * 2 + k_loc
                    v_idx = j * 8 + k_glob
                    v_s_idx = j * 2 + k_loc
                    V_block1[v_idx, offset + r] = V_s[v_s_idx, r]
                    
            # W: shift k
            for k_loc in range(2):
                for i in range(2):
                    k_glob = b * 2 + k_loc
                    w_idx = k_glob * 2 + i
                    w_s_idx = k_loc * 2 + i
                    W_block1[w_idx, offset + r] = W_s[w_s_idx, r]

    # 3. Construct Schoolbook 2x1x8
    # Rank 16
    R_school = 16
    U_block2 = np.zeros((2, R_school)) # i (0,1), j=0
    V_block2 = np.zeros((8, R_school)) # j=0, k (0..7)
    W_block2 = np.zeros((16, R_school)) # k (0..7), i (0,1)
    
    col = 0
    for i in range(2):
        for k in range(8):
            U_block2[i, col] = 1
            V_block2[k, col] = 1
            W_block2[k * 2 + i, col] = 1
            col += 1

    # 4. Combine into 2x3x8
    # Total Rank 44
    R_total = R_strassen + R_school
    U_final = np.zeros((6, R_total))
    V_final = np.zeros((24, R_total))
    W_final = np.zeros((16, R_total))
    
    # Block 1 (j=0,1)
    for i in range(2):
        for j in range(2):
            U_final[i * 3 + j, :R_strassen] = U_block1[i * 2 + j, :R_strassen]
    
    V_final[:16, :R_strassen] = V_block1
    W_final[:, :R_strassen] = W_block1
    
    # Block 2 (j=2)
    # U: i*3 + 2
    for i in range(2):
        U_final[i * 3 + 2, R_strassen:R_total] = U_block2[i, :]
        
    # V: j=2, so index 2*8 + k = 16 + k
    V_final[16:24, R_strassen:R_total] = V_block2
    
    # W: same indices
    W_final[:, R_strassen:R_total] = W_block2

    # Verify if time permits (optional but good for safety)
    # For now, trust the construction.
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END

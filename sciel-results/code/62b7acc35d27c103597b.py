"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a rank 16 decomposition by splitting the p=4 dimension into two p=2 blocks.
    # Each block is a 2x3x2 multiplication, decomposed into rank 8 (Strassen 7 + Outer 1).
    
    # Helper: Construct rank 8 decomposition for 2x3x2
    def construct_2x3x2():
        # Strassen for 2x2x2 (indices i,j in 0,1 for A; j,k in 0,1 for B; k,i in 0,1 for W)
        # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (a00 + a11)(b00 + b11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1 # C00, C11
        
        # M2 = (a10 + a11)b00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = -1 # C10, C11(-)
        
        # M3 = a00(b01 - b11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1 # C01, C11
        
        # M4 = a11(b10 - b00)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1 # C00, C10
        
        # M5 = (a00 + a01)b11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1 # C00(-), C01
        
        # M6 = (a10 - a00)(b00 + b01)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1 # C11
        
        # M7 = (a01 - a11)(b10 + b11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1 # C00
        
        # Outer product for j=2 (a is 2x1, b is 1x2)
        # Term: a_i2 * b_2k
        # U: rows i*3+2. For 2x3x2, rows are i*2+j? No, in 2x3x2, n=2, m=3, p=2.
        # U shape (6, 8). Rows i*3+j.
        # V shape (6, 8). Rows j*2+k.
        # W shape (4, 8). Rows k*2+i.
        
        U_1 = np.zeros((6, 8))
        V_1 = np.zeros((6, 8))
        W_1 = np.zeros((4, 8))
        
        # Embed Strassen (j=0,1) into U_1, V_1
        # U_s rows: i*2+j (j in 0,1). U_1 rows: i*3+j.
        # i=0, j=0 -> U_s[0], U_1[0]
        # i=0, j=1 -> U_s[1], U_1[1]
        # i=1, j=0 -> U_s[2], U_1[3]
        # i=1, j=1 -> U_s[3], U_1[4]
        
        U_1[0, 0:7] = U_s[0, :]
        U_1[1, 0:7] = U_s[1, :]
        U_1[3, 0:7] = U_s[2, :]
        U_1[4, 0:7] = U_s[3, :]
        
        # V_s rows: j*2+k (j in 0,1). V_1 rows: j*2+k (p=2).
        # j=0, k=0 -> V_s[0], V_1[0]
        # j=0, k=1 -> V_s[1], V_1[1]
        # j=1, k=0 -> V_s[2], V_1[2]
        # j=1, k=1 -> V_s[3], V_1[3]
        
        V_1[0, 0:7] = V_s[0, :]
        V_1[1, 0:7] = V_s[1, :]
        V_1[2, 0:7] = V_s[2, :]
        V_1[3, 0:7] = V_s[3, :]
        
        # W_s rows: k*2+i. W_1 rows: k*2+i.
        W_1[:, 0:7] = W_s[:, :]
        
        # Outer product (j=2)
        # U rows for j=2: i*3+2. i=0 -> 2, i=1 -> 5.
        U_1[2, 7] = 1
        U_1[5, 7] = 1
        
        # V rows for j=2: j*2+k. j=2. k=0 -> 4, k=1 -> 5.
        V_1[4, 7] = 1
        V_1[5, 7] = 1
        
        # W rows: k*2+i. All 1s.
        W_1[:, 7] = 1
        
        return U_1, V_1, W_1

    U_block, V_block, W_block = construct_2x3x2()
    
    # Combine two blocks for p=4
    # U_full: (6, 16)
    # V_full: (12, 16)
    # W_full: (8, 16)
    
    U_full = np.zeros((6, 16))
    V_full = np.zeros((12, 16))
    W_full = np.zeros((8, 16))
    
    # Block 1: k in 0,1
    U_full[:, 0:8] = U_block
    
    # V_block rows: j*2+k (k in 0,1).
    # V_full rows: j*4+k (k in 0,3).
    # For k=0,1:
    # j=0, k=0 -> V_full[0], V_block[0]
    # j=0, k=1 -> V_full[1], V_block[1]
    # j=1, k=0 -> V_full[4], V_block[2]
    # j=1, k=1 -> V_full[5], V_block[3]
    # j=2, k=0 -> V_full[8], V_block[4]
    # j=2, k=1 -> V_full[9], V_block[5]
    
    V_full[0, 0:8] = V_block[0, :]
    V_full[1, 0:8] = V_block[1, :]
    V_full[4, 0:8] = V_block[2, :]
    V_full[5, 0:8] = V_block[3, :]
    V_full[8, 0:8] = V_block[4, :]
    V_full[9, 0:8] = V_block[5, :]
    
    # W_block rows: k*2+i (k in 0,1).
    # W_full rows: k*2+i (k in 0,3).
    # For k=0,1:
    # k=0, i=0 -> W_full[0], W_block[0]
    # k=0, i=1 -> W_full[1], W_block[1]
    # k=1, i=0 -> W_full[2], W_block[2]
    # k=1, i=1 -> W_full[3], W_block[3]
    
    W_full[0:4, 0:8] = W_block[:, :]
    
    # Block 2: k in 2,3
    U_full[:, 8:16] = U_block
    
    # V_full rows for k=2,3:
    # j=0, k=2 -> V_full[2], V_block[0] (k=0 in block)
    # j=0, k=3 -> V_full[3], V_block[1] (k=1 in block)
    # j=1, k=2 -> V_full[6], V_block[2]
    # j=1, k=3 -> V_full[7], V_block[3]
    # j=2, k=2 -> V_full[10], V_block[4]
    # j=2, k=3 -> V_full[11], V_block[5]
    
    V_full[2, 8:16] = V_block[0, :]
    V_full[3, 8:16] = V_block[1, :]
    V_full[6, 8:16] = V_block[2, :]
    V_full[7, 8:16] = V_block[3, :]
    V_full[10, 8:16] = V_block[4, :]
    V_full[11, 8:16] = V_block[5, :]
    
    # W_full rows for k=2,3:
    # k=2, i=0 -> W_full[4], W_block[0]
    # k=2, i=1 -> W_full[5], W_block[1]
    # k=3, i=0 -> W_full[6], W_block[2]
    # k=3, i=1 -> W_full[7], W_block[3]
    
    W_full[4:8, 8:16] = W_block[:, :]
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END

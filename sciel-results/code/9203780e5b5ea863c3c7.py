import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 30 decomposition using Strassen for 2x2x2 blocks
    rank = 30
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen decomposition for 2x2x2
    # U_s, V_s, W_s are 4x7 matrices
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # Col 0: M1
    U_s[0, 0] = 1; U_s[3, 0] = 1  # A11, A22
    V_s[0, 0] = 1; V_s[3, 0] = 1  # B11, B22
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1 # C11+=, C12+=, C21+=, C22+=
    # Wait, C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # Let's fill W_s carefully based on standard Strassen
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # Reset W_s
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1
    W_s[0, 0] = 1  # C11
    W_s[3, 0] = 1  # C22
    
    # M2
    W_s[2, 1] = 1  # C21
    
    # M3
    W_s[1, 2] = 1  # C12
    W_s[3, 2] = -1 # C22
    
    # M4
    W_s[0, 3] = -1 # C11
    W_s[2, 3] = 1  # C21
    
    # M5
    W_s[1, 4] = 1  # C12
    W_s[3, 4] = -1 # C22
    
    # M6
    W_s[0, 5] = -1 # C11
    W_s[3, 5] = 1  # C22
    
    # M7
    W_s[0, 6] = 1  # C11
    
    # Check U_s and V_s for each M
    # M1: (A11+A22)(B11+B22)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    
    # M2: (A21+A22)B11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    
    # M3: A11(B12-B22)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    
    # M4: A22(B21-B11)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    
    # M5: (A11+A12)B22
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1

    # Helper to copy a 2x2x2 block into the global matrices
    def copy_block(start_col, a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # a_row_offset, a_col_offset: offset in A (2x4)
        # b_row_offset, b_col_offset: offset in B (4x5)
        # The block is 2x2x2.
        # U indices: i in 0..1, j in 0..1 -> global i*4 + (j+a_col_offset)
        # V indices: j in 0..1, k in 0..1 -> global (j+b_row_offset)*5 + (k+b_col_offset)
        # W indices: k in 0..1, i in 0..1 -> global (k+b_col_offset)*2 + i
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, r] != 0:
                        U[i*4 + (j+a_col_offset), start_col + r] += U_s[i*2+j, r]
            
            for j in range(2):
                for k in range(2):
                    if V_s[j*2+k, r] != 0:
                        V[(j+b_row_offset)*5 + (k+b_col_offset), start_col + r] += V_s[j*2+k, r]
            
            for k in range(2):
                for i in range(2):
                    if W_s[k*2+i, r] != 0:
                        W[(k+b_col_offset)*2 + i, start_col + r] += W_s[k*2+i, r]

    # Block 1: A_L (cols 0-1), B_T1 (rows 0-1, cols 0-1)
    copy_block(0, 0, 0, 0, 0)
    
    # Block 2: A_L (cols 0-1), B_T2 (rows 0-1, cols 2-3)
    copy_block(7, 0, 0, 0, 2)
    
    # Block 3: A_R (cols 2-3), B_B1 (rows 2-3, cols 0-1)
    copy_block(14, 0, 2, 2, 0)
    
    # Block 4: A_R (cols 2-3), B_B2 (rows 2-3, cols 2-3)
    copy_block(21, 0, 2, 2, 2)
    
    # Remaining 2 columns for the 5th column of B (k=4)
    # C_i4 = sum_j A_ij B_j4
    # Col 28: i=0
    for j in range(4):
        U[0*4 + j, 28] = 1
        V[j*5 + 4, 28] = 1
    W[4*2 + 0, 28] = 1
    
    # Col 29: i=1
    for j in range(4):
        U[1*4 + j, 29] = 1
        V[j*5 + 4, 29] = 1
    W[4*2 + 1, 29] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

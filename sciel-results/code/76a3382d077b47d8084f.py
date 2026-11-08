import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 98 solution: Two independent 4x4x4 Strassen decompositions
    # One for columns 0-3, one for columns 4-6 (padded to 7)
    
    # Rank 7 decomposition of 2x2x2
    U7 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V7 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ])
    W7 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 1]
    ])
    
    # Define Strassen mappings for 4x4x4
    # A blocks: A00, A01, A10, A11 (indices 0-3 in 2x2 block grid)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # M_A[k] is 4x16, maps vec(A) to vec(L_k(A))
    # vec(A) order: A00_00, A00_01, A00_10, A00_11, A01_00, ...
    # Actually, let's use block indices.
    # A is 4x4. Blocks are 2x2.
    # A_block[i,j] is A[2i:2i+2, 2j:2j+2].
    
    # We define the linear combinations for L_k(A) and R_k(B)
    # L_k(A) is a 2x2 matrix.
    # We represent L_k(A) as a vector of 4 entries.
    
    # k=1: L1 = A00 + A11
    # k=2: L2 = A10 + A11
    # k=3: L3 = A00
    # k=4: L4 = A11
    # k=5: L5 = A00 + A01
    # k=6: L6 = A10 - A00
    # k=7: L7 = A01 - A11
    
    # R_k(B)
    # k=1: R1 = B00 + B11
    # k=2: R2 = B00
    # k=3: R3 = B01 - B11
    # k=4: R4 = B10 - B00
    # k=5: R5 = B11
    # k=6: R6 = B00 + B01
    # k=7: R7 = B10 + B11
    
    # C_k contribution
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We build U, V, W for 4x4x4
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Helper to extract block from 4x4 matrix vector
    def get_block_indices(block_r, block_c):
        # Returns slice for 2x2 block
        return (slice(2*block_r, 2*block_r+2), slice(2*block_c, 2*block_c+2))
        
    # We need to map the 4 entries of L_k(A) to the 16 entries of A
    # L_k(A) is 2x2.
    # Let's define M_A[k] as 4x16.
    # M_A[k][i, j] = 1 if the i-th entry of L_k(A) comes from the j-th entry of A.
    
    # Entry (r,c) of L_k(A) is a linear combo of A blocks.
    # Let's just hardcode the 16x4 matrix for each k.
    
    # A indices: 0-3 (A00), 4-7 (A01), 8-11 (A10), 12-15 (A11)
    
    M_A = []
    M_B = []
    M_C = []
    
    # k=1
    # L1 = A00 + A11
    # vec(L1) = vec(A00) + vec(A11)
    M_A1 = np.zeros((4, 16))
    M_A1[0, 0] = 1; M_A1[1, 1] = 1; M_A1[2, 2] = 1; M_A1[3, 3] = 1
    M_A1[0, 12] = 1; M_A1[1, 13] = 1; M_A1[2, 14] = 1; M_A1[3, 15] = 1
    M_A.append(M_A1)
    
    # R1 = B00 + B11
    M_B1 = np.zeros((4, 16))
    M_B1[0, 0] = 1; M_B1[1, 1] = 1; M_B1[2, 2] = 1; M_B1[3, 3] = 1
    M_B1[0, 12] = 1; M_B1[1, 13] = 1; M_B1[2, 14] = 1; M_B1[3, 15] = 1
    M_B.append(M_B1)
    
    # C1 = M1 (contributes to C00, C11)
    # C00 += M1, C11 += M1
    # vec(C) indices: 0-3 (C00), 4-7 (C01), 8-11 (C10), 12-15 (C11)
    M_C1 = np.zeros((4, 16))
    M_C1[0, 0] = 1; M_C1[1, 1] = 1; M_C1[2, 2] = 1; M_C1[3, 3] = 1
    M_C1[0, 12] = 1; M_C1[1, 13] = 1; M_C1[2, 14] = 1; M_C1[3, 15] = 1
    M_C.append(M_C1)
    
    # k=2
    # L2 = A10 + A11
    M_A2 = np.zeros((4, 16))
    M_A2[0, 8] = 1; M_A2[1, 9] = 1; M_A2[2, 10] = 1; M_A2[3, 11] = 1
    M_A2[0, 12] = 1; M_A2[1, 13] = 1; M_A2[2, 14] = 1; M_A2[3, 15] = 1
    M_A.append(M_A2)
    
    # R2 = B00
    M_B2 = np.zeros((4, 16))
    M_B2[0, 0] = 1; M_B2[1, 1] = 1; M_B2[2, 2] = 1; M_B2[3, 3] = 1
    M_B.append(M_B2)
    
    # C2 = M2 (contributes to C10, C11)
    # C10 += M2, C11 -= M2
    M_C2 = np.zeros((4, 16))
    M_C2[0, 8] = 1; M_C2[1, 9] = 1; M_C2[2, 10] = 1; M_C2[3, 11] = 1
    M_C2[0, 12] = -1; M_C2[1, 13] = -1; M_C2[2, 14] = -1; M_C2[3, 15] = -1
    M_C.append(M_C2)
    
    # k=3
    # L3 = A00
    M_A3 = np.zeros((4, 16))
    M_A3[0, 0] = 1; M_A3[1, 1] = 1; M_A3[2, 2] = 1; M_A3[3, 3] = 1
    M_A.append(M_A3)
    
    # R3 = B01 - B11
    M_B3 = np.zeros((4, 16))
    M_B3[0, 4] = 1; M_B3[1, 5] = 1; M_B3[2, 6] = 1; M_B3[3, 7] = 1
    M_B3[0, 12] = -1; M_B3[1, 13] = -1; M_B3[2, 14] = -1; M_B3[3, 15] = -1
    M_B.append(M_B3)
    
    # C3 = M3 (contributes to C01, C11)
    # C01 += M3, C11 += M3
    M_C3 = np.zeros((4, 16))
    M_C3[0, 4] = 1; M_C3[1, 5] = 1; M_C3[2, 6] = 1; M_C3[3, 7] = 1
    M_C3[0, 12] = 1; M_C3[1, 13] = 1; M_C3[2, 14] = 1; M_C3[3, 15] = 1
    M_C.append(M_C3)
    
    # k=4
    # L4 = A11
    M_A4 = np.zeros((4, 16))
    M_A4[0, 12] = 1; M_A4[1, 13] = 1; M_A4[2, 14] = 1; M_A4[3, 15] = 1
    M_A.append(M_A4)
    
    # R4 = B10 - B00
    M_B4 = np.zeros((4, 16))
    M_B4[0, 8] = 1; M_B4[1, 9] = 1; M_B4[2, 10] = 1; M_B4[3, 11] = 1
    M_B4[0, 0] = -1; M_B4[1, 1] = -1; M_B4[2, 2] = -1; M_B4[3, 3] = -1
    M_B.append(M_B4)
    
    # C4 = M4 (contributes to C00, C10)
    # C00 += M4, C10 += M4
    M_C4 = np.zeros((4, 16))
    M_C4[0, 0] = 1; M_C4[1, 1] = 1; M_C4[2, 2] = 1; M_C4[3, 3] = 1
    M_C4[0, 8] = 1; M_C4[1, 9] = 1; M_C4[2, 10] = 1; M_C4[3, 11] = 1
    M_C.append(M_C4)
    
    # k=5
    # L5 = A00 + A01
    M_A5 = np.zeros((4, 16))
    M_A5[0, 0] = 1; M_A5[1, 1] = 1; M_A5[2, 2] = 1; M_A5[3, 3] = 1
    M_A5[0, 4] = 1; M_A5[1, 5] = 1; M_A5[2, 6] = 1; M_A5[3, 7] = 1
    M_A.append(M_A5)
    
    # R5 = B11
    M_B5 = np.zeros((4, 16))
    M_B5[0, 12] = 1; M_B5[1, 13] = 1; M_B5[2, 14] = 1; M_B5[3, 15] = 1
    M_B.append(M_B5)
    
    # C5 = M5 (contributes to C00, C01)
    # C00 -= M5, C01 += M5
    M_C5 = np.zeros((4, 16))
    M_C5[0, 0] = -1; M_C5[1, 1] = -1; M_C5[2, 2] = -1; M_C5[3, 3] = -1
    M_C5[0, 4] = 1; M_C5[1, 5] = 1; M_C5[2, 6] = 1; M_C5[3, 7] = 1
    M_C.append(M_C5)
    
    # k=6
    # L6 = A10 - A00
    M_A6 = np.zeros((4, 16))
    M_A6[0, 8] = 1; M_A6[1, 9] = 1; M_A6[2, 10] = 1; M_A6[3, 11] = 1
    M_A6[0, 0] = -1; M_A6[1, 1] = -1; M_A6[2, 2] = -1; M_A6[3, 3] = -1
    M_A.append(M_A6)
    
    # R6 = B00 + B01
    M_B6 = np.zeros((4, 16))
    M_B6[0, 0] = 1; M_B6[1, 1] = 1; M_B6[2, 2] = 1; M_B6[3, 3] = 1
    M_B6[0, 4] = 1; M_B6[1, 5] = 1; M_B6[2, 6] = 1; M_B6[3, 7] = 1
    M_B.append(M_B6)
    
    # C6 = M6 (contributes to C11)
    # C11 += M6
    M_C6 = np.zeros((4, 16))
    M_C6[0, 12] = 1; M_C6[1, 13] = 1; M_C6[2, 14] = 1; M_C6[3, 15] = 1
    M_C.append(M_C6)
    
    # k=7
    # L7 = A01 - A11
    M_A7 = np.zeros((4, 16))
    M_A7[0, 4] = 1; M_A7[1, 5] = 1; M_A7[2, 6] = 1; M_A7[3, 7] = 1
    M_A7[0, 12] = -1; M_A7[1, 13] = -1; M_A7[2, 14] = -1; M_A7[3, 15] = -1
    M_A.append(M_A7)
    
    # R7 = B10 + B11
    M_B7 = np.zeros((4, 16))
    M_B7[0, 8] = 1; M_B7[1, 9] = 1; M_B7[2, 10] = 1; M_B7[3, 11] = 1
    M_B7[0, 12] = 1; M_B7[1, 13] = 1; M_B7[2, 14] = 1; M_B7[3, 15] = 1
    M_B.append(M_B7)
    
    # C7 = M7 (contributes to C00)
    # C00 += M7
    M_C7 = np.zeros((4, 16))
    M_C7[0, 0] = 1; M_C7[1, 1] = 1; M_C7[2, 2] = 1; M_C7[3, 3] = 1
    M_C.append(M_C7)
    
    # Build U_444, V_444, W_444
    for k in range(7):
        # U_k = M_A[k].T @ U7
        U_444[:, k*7:(k+1)*7] = M_A[k].T @ U7
        V_444[:, k*7:(k+1)*7] = M_B[k].T @ V7
        W_444[:, k*7:(k+1)*7] = M_C[k].T @ W7
        
    # Now we have rank 49 for 4x4x4.
    # We need rank for 4x4x7.
    # Use two 4x4x4 decompositions.
    # One for cols 0-3, one for cols 4-7 (col 7 is 0).
    
    U_total = np.zeros((16, 98))
    V_total = np.zeros((16, 98))
    W_total = np.zeros((28, 98)) # 4x7 -> 28 entries
    
    # First 49 columns: cols 0-3
    U_total[:, 0:49] = U_444
    # V for cols 0-3: B blocks B00, B01 (top), B10, B11 (bot)
    # B is 4x4. B00 is top-left, B01 is top-right, etc.
    # In our 4x4x4, B is 4x4.
    # Here B is 4x7. We take B[:, 0:4] for the first part.
    # The V_444 assumes B is 4x4.
    # V_444 rows correspond to B entries.
    # We just use V_444.
    V_total[:, 0:49] = V_444
    
    # W for cols 0-3: C is 4x4.
    # W_444 rows correspond to C entries.
    # C entries 0-15 correspond to C[:, 0:4].
    W_total[0:16, 0:49] = W_444
    
    # Second 49 columns: cols 4-7
    # B[:, 4:8] (col 7 is 0)
    U_total[:, 49:98] = U_444
    V_total[:, 49:98] = V_444
    # C entries 16-31 correspond to C[:, 4:8].
    W_total[16:28, 49:98] = W_444
    
    # Convert to lists
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

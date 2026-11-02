"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the problem into two 2x2x4 multiplications.
    # A = [A_L, A_R] where A_L, A_R are 2x2.
    # B = [B_T; B_B] where B_T, B_B are 2x4.
    # C = A_L B_T + A_R B_B.
    
    # We need a rank decomposition for 2x2x4 matrix multiplication.
    # Let A be 2x2, B be 2x4, C be 2x4.
    # C = A B.
    # Rank 14 decomposition.
    
    # We will define the 14 terms for a single 2x2x4 block.
    # U_block: shape (2*2, 14) = (4, 14)
    # V_block: shape (2*4, 14) = (8, 14)
    # W_block: shape (4*2, 14) = (8, 14)
    
    # Using standard Strassen-like extension or known optimal.
    # For 2x2x2, we have 7 terms.
    # For 2x2x4, we can use two 2x2x2 blocks sharing A.
    # This yields 14 terms.
    
    # Let's define the 14 terms explicitly for 2x2x4.
    # A indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # B indices: (0,0)..(1,3) -> flat: 0..7
    # C indices: (0,0)..(1,3) -> flat: 0..7 (W[k*n+i] = W[k*2+i])
    
    U_block = [[0]*14 for _ in range(4)]
    V_block = [[0]*14 for _ in range(8)]
    W_block = [[0]*14 for _ in range(8)]
    
    # Term 1-7: For B_T (cols 0,1 of B, which are rows 0,1 of the 2x4 B? No, B is 2x4.
    # B_T is the first 2 columns? No, B is 2x4.
    # In the sub-problem 2x2x4, B is 2x4.
    # Let's split B into two 2x2 matrices: B_1 (cols 0,1) and B_2 (cols 2,3).
    # C_1 = A B_1 (2x2), C_2 = A B_2 (2x2).
    
    # Strassen for A B_1:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Strassen for A B_2:
    # N1 = (A00+A11)(B20+B31)  # Note: B_2 indices are (0,2),(0,3),(1,2),(1,3)
    # Let's map B_2 local (r,c) to global (r, c+2).
    # N1 = (A00+A11)(B[0,2]+B[1,3])
    # N2 = (A10+A11)B[0,2]
    # N3 = A00(B[0,3]-B[1,3])
    # N4 = A11(B[1,2]-B[0,2])
    # N5 = (A00+A01)B[1,3]
    # N6 = (A10-A00)(B[0,2]+B[0,3])
    # N7 = (A01-A11)(B[1,2]+B[1,3])
    
    # Fill U_block (same for both sets of terms)
    # A flat indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Terms 0-6 (for B_1)
    U_block[0][0] += 1; U_block[3][0] += 1 # M1: A00, A11
    U_block[2][1] += 1; U_block[3][1] += 1 # M2: A10, A11
    U_block[0][2] += 1                     # M3: A00
    U_block[3][3] += 1                     # M4: A11
    U_block[0][4] += 1; U_block[1][4] += 1 # M5: A00, A01
    U_block[2][5] += 1; U_block[0][5] += -1# M6: A10, -A00
    U_block[1][6] += 1; U_block[3][6] += -1# M7: A01, -A11
    
    # Terms 7-13 (for B_2)
    U_block[0][7] += 1; U_block[3][7] += 1 # N1: A00, A11
    U_block[2][8] += 1; U_block[3][8] += 1 # N2: A10, A11
    U_block[0][9] += 1                     # N3: A00
    U_block[3][10] += 1                    # N4: A11
    U_block[0][11] += 1; U_block[1][11] += 1# N5: A00, A01
    U_block[2][12] += 1; U_block[0][12] += -1# N6: A10, -A00
    U_block[1][13] += 1; U_block[3][13] += -1# N7: A01, -A11
    
    # Fill V_block
    # B flat indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(0,3), 6:(1,2), 7:(1,3)
    
    # Terms 0-6 (for B_1)
    V_block[0][0] += 1; V_block[3][0] += 1 # M1: B00, B11
    V_block[0][1] += 1                     # M2: B00
    V_block[1][2] += 1; V_block[3][2] += -1# M3: B01, -B11
    V_block[2][3] += 1; V_block[0][3] += -1# M4: B10, -B00
    V_block[3][4] += 1                     # M5: B11
    V_block[0][5] += 1; V_block[1][5] += 1 # M6: B00, B01
    V_block[2][6] += 1; V_block[3][6] += 1 # M7: B10, B11
    
    # Terms 7-13 (for B_2)
    V_block[4][7] += 1; V_block[7][7] += 1 # N1: B[0,2], B[1,3]
    V_block[4][8] += 1                     # N2: B[0,2]
    V_block[5][9] += 1; V_block[7][9] += -1# N3: B[0,3], -B[1,3]
    V_block[6][10] += 1; V_block[4][10] += -1# N4: B[1,2], -B[0,2]
    V_block[7][11] += 1                    # N5: B[1,3]
    V_block[4][12] += 1; V_block[5][12] += 1# N6: B[0,2], B[0,3]
    V_block[6][13] += 1; V_block[7][13] += 1# N7: B[1,2], B[1,3]
    
    # Fill W_block
    # C flat indices (W[k*n+i]): 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2), 6:(0,3), 7:(1,3)
    # Wait, W index is k*n+i.
    # C_1 is cols 0,1. C_2 is cols 2,3.
    
    # Terms 0-6 (for C_1)
    # M1 -> C00, C11
    W_block[0][0] += 1; W_block[3][0] += 1
    # M2 -> C10, C11 (with -1 on C11? No, Strassen W:
    # C00 = M1 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5
    # Let's re-verify Strassen W coefficients.
    # C00 = M1 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5
    
    # M1: C00(+1), C11(+1)
    W_block[0][0] += 1; W_block[3][0] += 1
    # M2: C10(+1)
    W_block[1][1] += 1 # Index for C10 is 1*2+0 = 2? No.
    # W index = k*n + i.
    # C10: k=1, i=0 -> 1*2+0 = 2.
    # C00: k=0, i=0 -> 0.
    # C01: k=0, i=1 -> 1.
    # C11: k=1, i=1 -> 3.
    
    # Let's redo W filling carefully.
    # W_block is 8x14.
    # Indices:
    # 0: C00
    # 1: C01
    # 2: C10
    # 3: C11
    # 4: C02
    # 5: C03
    # 6: C12
    # 7: C13
    
    # Clear previous W_block and refill
    W_block = [[0]*14 for _ in range(8)]
    
    # M1 (term 0): C00+1, C11+1
    W_block[0][0] += 1; W_block[3][0] += 1
    # M2 (term 1): C10+1
    W_block[2][1] += 1
    # M3 (term 2): C11-1
    W_block[3][2] += -1
    # M4 (term 3): C00-1, C10+1
    W_block[0][3] += -1; W_block[2][3] += 1
    # M5 (term 4): C01+1, C11+1
    W_block[1][4] += 1; W_block[3][4] += 1
    # M6 (term 5): C00+1
    W_block[0][5] += 1
    # M7 (term 6): C01+1
    W_block[1][6] += 1
    
    # N1 (term 7): C02+1, C13+1
    W_block[4][7] += 1; W_block[7][7] += 1
    # N2 (term 8): C12+1
    W_block[6][8] += 1
    # N3 (term 9): C13-1
    W_block[7][9] += -1
    # N4 (term 10): C02-1, C12+1
    W_block[4][10] += -1; W_block[6][10] += 1
    # N5 (term 11): C03+1, C13+1
    W_block[5][11] += 1; W_block[7][11] += 1
    # N6 (term 12): C02+1
    W_block[4][12] += 1
    # N7 (term 13): C03+1
    W_block[5][13] += 1

    # Now map to 2x4x4.
    # Rank = 14 + 14 = 28.
    rank = 28
    U = [[0]*rank for _ in range(n*m)] # 8x28
    V = [[0]*rank for _ in range(m*p)] # 16x28
    W = [[0]*rank for _ in range(p*n)] # 8x28
    
    # Block 1: A_L (cols 0,1) * B_T (rows 0,1) -> C (cols 0,1)
    # A_L flat indices: 0,1,2,3
    # B_T flat indices: 0..7 (rows 0,1 of B)
    # C_L flat indices: 0..7 (cols 0,1 of C)
    
    for r in range(14):
        # U
        for i in range(4):
            if U_block[i][r] != 0:
                U[i][r] = U_block[i][r]
        # V
        for j in range(8):
            if V_block[j][r] != 0:
                V[j][r] = V_block[j][r]
        # W
        for k in range(8):
            if W_block[k][r] != 0:
                W[k][r] = W_block[k][r]
                
    # Block 2: A_R (cols 2,3) * B_B (rows 2,3) -> C (cols 2,3)
    # A_R flat indices: 4,5,6,7
    # B_B flat indices: 8..15 (rows 2,3 of B)
    # C_R flat indices: 4..7 (cols 2,3 of C)
    
    for r in range(14):
        r_offset = 14 + r
        # U
        for i in range(4):
            if U_block[i][r] != 0:
                U[i+4][r_offset] = U_block[i][r]
        # V
        for j in range(8):
            if V_block[j][r] != 0:
                V[j+8][r_offset] = V_block[j][r]
        # W
        for k in range(8):
            if W_block[k][r] != 0:
                W[k+4][r_offset] = W_block[k][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

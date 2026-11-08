import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank_1 = 4 * 8 # 32
    rank_2 = 8 * 7 # 56
    total_rank = rank_1 + rank_2
    
    U = [[0] * total_rank for _ in range(n * m)] # 12 x 88
    V = [[0] * total_rank for _ in range(m * p)] # 32 x 88
    W = [[0] * total_rank for _ in range(p * n)] # 24 x 88
    
    # Part 1: 1x4 by 4x8 (Standard)
    # n1=1, m1=4, p1=8
    # A1 is row 0 of A.
    # B is full B.
    # C1 is row 0 of C.
    
    col = 0
    for j in range(4):
        for k in range(8):
            # U index: i*m + j. i=0.
            U[0 * 4 + j][col] = 1
            # V index: j*p + k.
            V[j * 8 + k][col] = 1
            # W index: k*n + i. i=0.
            W[k * 3 + 0][col] = 1
            col += 1
            
    # Part 2: 2x4 by 4x8 (Strassen)
    # n2=2, m2=4, p2=8
    # A2 is rows 1,2 of A.
    # B is full B.
    # C2 is rows 1,2 of C.
    
    # Strassen basis for 2x2 x 2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    # P1: (A00+A11)(B00+B11)
    U_S[0][0] = 1; U_S[3][0] = 1
    V_S[0][0] = 1; V_S[3][0] = 1
    W_S[0][0] = 1; W_S[3][0] = 1; W_S[1][0] = -1; W_S[2][0] = -1
    
    # P2: (A10+A11)B00
    U_S[2][1] = 1; U_S[3][1] = 1
    V_S[0][1] = 1
    W_S[2][1] = 1; W_S[3][1] = -1
    
    # P3: A00(B01-B11)
    U_S[0][2] = 1
    V_S[1][2] = 1; V_S[3][2] = -1
    W_S[1][2] = 1; W_S[3][2] = 1
    
    # P4: A11(B10-B00)
    U_S[3][3] = 1
    V_S[2][3] = 1; V_S[0][3] = -1
    W_S[0][3] = 1; W_S[2][3] = 1
    
    # P5: (A00+A01)B11
    U_S[0][4] = 1; U_S[1][4] = 1
    V_S[3][4] = 1
    W_S[0][4] = -1; W_S[1][4] = 1
    
    # P6: (A10-A00)(B00+B01)
    U_S[2][5] = 1; U_S[0][5] = -1
    V_S[0][5] = 1; V_S[1][5] = 1
    W_S[3][5] = 1
    
    # P7: (A01-A11)(B10+B11)
    U_S[1][6] = 1; U_S[3][6] = -1
    V_S[2][6] = 1; V_S[3][6] = 1
    W_S[0][6] = 1; W_S[1][6] = 1
    
    # We have 8 blocks to process.
    # Block 0: A_L * B_T0
    # Block 1: A_R * B_B0
    # Block 2: A_L * B_T1
    # Block 3: A_R * B_B1
    # ...
    
    for block_idx in range(8):
        # Determine which A part and which B part
        if block_idx % 2 == 0:
            # A_L
            a_offset = 1 * 4 # Row 1 of A is start of A_L?
            # A2 is rows 1,2 of global A.
            # A_L is first 2 cols of A2.
            # Global A indices:
            # Row 1: 1*4 + 0, 1*4 + 1 (A_L)
            # Row 2: 2*4 + 0, 2*4 + 1 (A_L)
            # So A_L local indices 0,1,2,3 map to global 4,5,8,9.
            a_map = [4, 5, 8, 9]
            
            # B part: B_T[block_idx // 2]
            # B_T is top 2 rows of B.
            # B_T0 is cols 0,1. B_T1 is cols 2,3, etc.
            b_col_offset = (block_idx // 2) * 2
            # B_T rows are global B rows 0,1.
            # B_T0: B[0,0], B[0,1], B[1,0], B[1,1]
            # Local B indices 0,1,2,3 map to:
            # 0: 0*8 + b_col_offset
            # 1: 0*8 + b_col_offset + 1
            # 2: 1*8 + b_col_offset
            # 3: 1*8 + b_col_offset + 1
            b_map = [0*8 + b_col_offset, 0*8 + b_col_offset + 1, 1*8 + b_col_offset, 1*8 + b_col_offset + 1]
            
            # C part: C[block_idx // 2]
            # C is 2x8.
            # C0 is cols 0,1.
            # C rows are global C rows 1,2 (since A2 is rows 1,2).
            # C_local 0,1,2,3 map to:
            # C row 1, col b_col_offset
            # C row 1, col b_col_offset + 1
            # C row 2, col b_col_offset
            # C row 2, col b_col_offset + 1
            # Global W index: k*n + i.
            # i is row of C (1 or 2). k is col of C.
            # W index for C[i, k] is k*3 + i.
            c_map = [
                b_col_offset * 3 + 1,
                (b_col_offset + 1) * 3 + 1,
                b_col_offset * 3 + 2,
                (b_col_offset + 1) * 3 + 2
            ]
        else:
            # A_R
            # A_R is last 2 cols of A2.
            # Global A indices:
            # Row 1: 1*4 + 2, 1*4 + 3
            # Row 2: 2*4 + 2, 2*4 + 3
            a_map = [6, 7, 10, 11]
            
            # B part: B_B[block_idx // 2]
            # B_B is bottom 2 rows of B.
            b_col_offset = (block_idx // 2) * 2
            # B_B rows are global B rows 2,3.
            b_map = [2*8 + b_col_offset, 2*8 + b_col_offset + 1, 3*8 + b_col_offset, 3*8 + b_col_offset + 1]
            
            # C part: C[block_idx // 2]
            # Same C mapping as above.
            c_map = [
                b_col_offset * 3 + 1,
                (b_col_offset + 1) * 3 + 1,
                b_col_offset * 3 + 2,
                (b_col_offset + 1) * 3 + 2
            ]
            
        # Fill U, V, W for this block
        for local_c in range(7):
            global_c = col + local_c
            for local_a in range(4):
                val = U_S[local_a][local_c]
                if val != 0:
                    U[a_map[local_a]][global_c] += val
            for local_b in range(4):
                val = V_S[local_b][local_c]
                if val != 0:
                    V[b_map[local_b]][global_c] += val
            for local_c_idx in range(4):
                val = W_S[local_c_idx][local_c]
                if val != 0:
                    W[c_map[local_c_idx]][global_c] += val
        col += 7
        
    return {"U": U, "V": V, "W": W}

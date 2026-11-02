import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Strassen 2x2x2 decomposition (Rank 7)
    # T_ijk = sum_c U[i,c] V[j,c] W[k,c]
    # U, V, W are 4x7
    U7 = [[0]*7 for _ in range(4)]
    V7 = [[0]*7 for _ in range(4)]
    W7 = [[0]*7 for _ in range(4)]
    
    # c=0: (00+11)(00+11) -> C00, C11
    # c=1: (10+11)00 -> C10
    # c=2: 00 10 -> C01
    # c=3: 11 11 -> C00, C01, C10, C11
    # c=4: (00+01)11 -> C11
    # c=5: (10-00)(00+01) -> C00
    # c=6: (01-11)(00+10) -> C11
    
    # Indices: A[i,j], B[j,k], C[i,k]
    # U[i, c], V[j, c], W[k, c]
    
    # c=0
    U7[0][0]=1; U7[1][0]=1; U7[2][0]=1; U7[3][0]=1
    V7[0][0]=1; V7[1][0]=1; V7[2][0]=1; V7[3][0]=1
    W7[0][0]=1; W7[1][0]=1; W7[2][0]=1; W7[3][0]=1
    
    # c=1
    U7[2][1]=1; U7[3][1]=1
    V7[0][1]=1; V7[1][1]=1
    W7[2][1]=1; W7[3][1]=1
    
    # c=2
    U7[0][2]=1; U7[1][2]=1
    V7[2][2]=1; V7[3][2]=1
    W7[0][2]=1; W7[1][2]=1
    
    # c=3
    U7[2][3]=1; U7[3][3]=1
    V7[2][3]=1; V7[3][3]=1
    W7[0][3]=1; W7[1][3]=1; W7[2][3]=1; W7[3][3]=1
    
    # c=4
    U7[0][4]=1; U7[1][4]=1; U7[2][4]=1; U7[3][4]=1
    V7[2][4]=1; V7[3][4]=1
    W7[2][4]=1; W7[3][4]=1
    
    # c=5
    U7[2][5]=1; U7[3][5]=1; U7[0][5]=-1; U7[1][5]=-1
    V7[0][5]=1; V7[1][5]=1; V7[2][5]=1; V7[3][5]=1
    W7[0][5]=1; W7[1][5]=1
    
    # c=6
    U7[0][6]=1; U7[1][6]=1; U7[2][6]=-1; U7[3][6]=-1
    V7[0][6]=1; V7[1][6]=1; V7[2][6]=1; V7[3][6]=1
    W7[2][6]=1; W7[3][6]=1

    # 2. Build 2x2x4 decomposition (Rank 14)
    # A is 2x2, B is 2x4, C is 2x4
    # U14: 4x14, V14: 8x14, W14: 8x14
    U14 = [[0]*14 for _ in range(4)]
    V14 = [[0]*14 for _ in range(8)]
    W14 = [[0]*14 for _ in range(8)]
    
    for c in range(7):
        # First half of B (cols 0,1)
        for i in range(4):
            U14[i][c] = U7[i][c]
        for j in range(4):
            V14[j][c] = V7[j][c]
            V14[j+4][c] = 0
        for k in range(4):
            W14[k][c] = W7[k][c]
            W14[k+4][c] = 0
            
        # Second half of B (cols 2,3)
        c2 = c + 7
        for i in range(4):
            U14[i][c2] = U7[i][c]
        for j in range(4):
            V14[j][c2] = 0
            V14[j+4][c2] = V7[j][c]
        for k in range(4):
            W14[k][c2] = 0
            W14[k+4][c2] = W7[k][c]

    # 3. Build 4x4x8 decomposition (Rank 98)
    # A is 4x4, B is 4x8, C is 4x8
    # U: 16x98, V: 32x98, W: 32x98
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # Strassen blocks for A (2x2 blocks of 2x2)
    # A00: rows 0,1 cols 0,1 -> U indices 0,1
    # A01: rows 0,1 cols 2,3 -> U indices 2,3
    # A10: rows 2,3 cols 0,1 -> U indices 4,5
    # A11: rows 2,3 cols 2,3 -> U indices 6,7
    
    # Strassen blocks for B (2x4 blocks)
    # B00: rows 0,1 cols 0..3 -> V indices 0..7
    # B01: rows 0,1 cols 4..7 -> V indices 8..15
    # B10: rows 2,3 cols 0..3 -> V indices 16..23
    # B11: rows 2,3 cols 4..7 -> V indices 24..31
    
    # Strassen products M_k
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00 B10
    # M3 = A11 B11
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B00+B10)
    
    # Contributions to C blocks
    # C00 = M0 - M3 + M5
    # C01 = M2 + M3
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4
    
    # C00: rows 0,1 cols 0..3 -> W indices 0..7
    # C01: rows 0,1 cols 4..7 -> W indices 8..15
    # C10: rows 2,3 cols 0..3 -> W indices 16..23
    # C11: rows 2,3 cols 4..7 -> W indices 24..31
    
    def add_block(c_offset, u_rows, v_rows, w_rows, sign=1):
        # u_rows: list of (local_u_row, global_u_row)
        # v_rows: list of (local_v_row, global_v_row)
        # w_rows: list of (local_w_row, global_w_row)
        for c_l in range(14):
            c = c_offset + c_l
            for lu, gu in u_rows:
                U[gu][c] += sign * U14[lu][c_l]
            for lv, gv in v_rows:
                V[gv][c] += sign * V14[lv][c_l]
            for lw, gw in w_rows:
                W[gw][c] += sign * W14[lw][c_l]

    # M0
    c_off = 0
    u_rows = [(0,0), (1,1), (0,4), (1,5), (0,6), (1,7)] # A00 rows 0,1 and A11 rows 2,3 mapped to 0,1,4,5,6,7? 
    # Wait, U14 is 4x14. The local U rows are 0,1,2,3 corresponding to the 2x2 A block.
    # A00 is rows 0,1 of A. So local 0,1 -> global 0,1.
    # A11 is rows 2,3 of A. So local 0,1 -> global 2,3.
    # So u_rows for A00+A11:
    # Local 0 -> Global 0
    # Local 1 -> Global 1
    # Local 0 -> Global 2
    # Local 1 -> Global 3
    u_rows_m0 = [(0,0), (1,1), (0,2), (1,3)]
    
    # B00+B11:
    # B00 is rows 0,1 of B. Local V rows 0,1,2,3 (for 2x4 block? No, V14 is 8x14).
    # V14 local rows 0..3 correspond to first 2x2 block of B (cols 0,1).
    # V14 local rows 4..7 correspond to second 2x2 block of B (cols 2,3).
    # Wait, B is 2x4. V14 is built for 2x4.
    # V14 rows 0..7.
    # Rows 0,1,2,3 correspond to B cols 0,1.
    # Rows 4,5,6,7 correspond to B cols 2,3.
    # So for B00 (rows 0,1 of B, cols 0..3):
    # Global V rows 0..7.
    # Local V rows 0..7.
    # So v_rows for B00: (0,0)...(7,7).
    
    # For B11 (rows 2,3 of B, cols 4..7):
    # Global V rows 24..31.
    # Local V rows 0..7.
    # So v_rows for B11: (0,24)...(7,31).
    
    v_rows_m0 = [(i, i) for i in range(8)] + [(i, 24+i) for i in range(8)]
    
    # W for M0:
    # M0 contributes to C00 and C11.
    # C00: W rows 0..7. Local W rows 0..7.
    # C11: W rows 24..31. Local W rows 0..7.
    w_rows_m0 = [(i, i) for i in range(8)] + [(i, 24+i) for i in range(8)]
    
    add_block(0, u_rows_m0, v_rows_m0, w_rows_m0, 1)

    # M1 = (A10+A11)B00
    c_off = 14
    # A10: rows 2,3 cols 0,1. Local 0,1 -> Global 4,5.
    # A11: rows 2,3 cols 2,3. Local 0,1 -> Global 6,7.
    u_rows_m1 = [(0,4), (1,5), (0,6), (1,7)]
    v_rows_m1 = [(i, i) for i in range(8)] # B00
    # M1 contributes to C10 (rows 2,3 cols 0..3). W rows 16..23.
    w_rows_m1 = [(i, 16+i) for i in range(8)]
    add_block(14, u_rows_m1, v_rows_m1, w_rows_m1, 1)

    # M2 = A00 B10
    c_off = 28
    # A00: Local 0,1 -> Global 0,1.
    u_rows_m2 = [(0,0), (1,1)]
    # B10: rows 2,3 cols 0..3. Global V 16..23.
    v_rows_m2 = [(i, 16+i) for i in range(8)]
    # M2 contributes to C01 (rows 0,1 cols 4..7). W rows 8..15.
    w_rows_m2 = [(i, 8+i) for i in range(8)]
    add_block(28, u_rows_m2, v_rows_m2, w_rows_m2, 1)

    # M3 = A11 B11
    c_off = 42
    # A11: Local 0,1 -> Global 2,3.
    u_rows_m3 = [(0,2), (1,3)]
    # B11: Global V 24..31.
    v_rows_m3 = [(i, 24+i) for i in range(8)]
    # M3 contributes to C00 (-), C01 (+), C10 (+), C11 (+).
    # We add M3 with sign 1 here, and handle signs in W accumulation?
    # No, add_block adds to W.
    # C00: W 0..7 (Sign -)
    # C01: W 8..15 (Sign +)
    # C10: W 16..23 (Sign +)
    # C11: W 24..31 (Sign +)
    # We can't do different signs in one add_block call easily.
    # Let's split M3 into 4 parts? No, U and V are same.
    # We can call add_block 4 times with different W rows and signs.
    
    # M3 -> C00 (Sign -)
    w_rows_m3_1 = [(i, i) for i in range(8)]
    add_block(42, u_rows_m3, v_rows_m3, w_rows_m3_1, -1)
    
    # M3 -> C01 (Sign +)
    w_rows_m3_2 = [(i, 8+i) for i in range(8)]
    add_block(42, u_rows_m3, v_rows_m3, w_rows_m3_2, 1)
    
    # M3 -> C10 (Sign +)
    w_rows_m3_3 = [(i, 16+i) for i in range(8)]
    add_block(42, u_rows_m3, v_rows_m3, w_rows_m3_3, 1)
    
    # M3 -> C11 (Sign +)
    w_rows_m3_4 = [(i, 24+i) for i in range(8)]
    add_block(42, u_rows_m3, v_rows_m3, w_rows_m3_4, 1)

    # M4 = (A00+A01)B11
    c_off = 56
    # A00: Local 0,1 -> Global 0,1.
    # A01: Local 0,1 -> Global 2,3.
    u_rows_m4 = [(0,0), (1,1), (0,2), (1,3)]
    v_rows_m4 = [(i, 24+i) for i in range(8)] # B11
    # M4 contributes to C11 (Sign +).
    w_rows_m4 = [(i, 24+i) for i in range(8)]
    add_block(56, u_rows_m4, v_rows_m4, w_rows_m4, 1)

    # M5 = (A10-A00)(B00+B01)
    c_off = 70
    # A10: Local 0,1 -> Global 4,5.
    # A00: Local 0,1 -> Global 0,1.
    u_rows_m5 = [(0,4), (1,5), (0,0), (1,1)]
    # Sign: A10 is +, A00 is -.
    # U14 local 0,1 get +1 for A10, -1 for A00.
    # So we need to modify U14 values or use two calls.
    # Let's use two calls.
    
    # Part 1: A10 (Sign +)
    u_rows_m5_p1 = [(0,4), (1,5)]
    # B00+B01: B00 (V 0..7), B01 (V 8..15).
    v_rows_m5 = [(i, i) for i in range(8)] + [(i, 8+i) for i in range(8)]
    # M5 contributes to C00 (Sign +).
    w_rows_m5 = [(i, i) for i in range(8)]
    add_block(70, u_rows_m5_p1, v_rows_m5, w_rows_m5, 1)
    
    # Part 2: -A00 (Sign -)
    u_rows_m5_p2 = [(0,0), (1,1)]
    add_block(70, u_rows_m5_p2, v_rows_m5, w_rows_m5, -1)

    # M6 = (A01-A11)(B00+B10)
    c_off = 84
    # A01: Local 0,1 -> Global 2,3.
    # A11: Local 0,1 -> Global 6,7.
    u_rows_m6_p1 = [(0,2), (1,3)] # A01
    u_rows_m6_p2 = [(0,6), (1,7)] # -A11
    # B00+B10: B00 (V 0..7), B10 (V 16..23).
    v_rows_m6 = [(i, i) for i in range(8)] + [(i, 16+i) for i in range(8)]
    # M6 contributes to C11 (Sign +).
    w_rows_m6 = [(i, 24+i) for i in range(8)]
    add_block(84, u_rows_m6_p1, v_rows_m6, w_rows_m6, 1)
    add_block(84, u_rows_m6_p2, v_rows_m6, w_rows_m6, -1)

    # Project to 4x4x7 (remove 8th column)
    # W shape is 32x98.
    # W index k*4+i.
    # k=0..6.
    # We keep rows 0..27 (k=0..6).
    # Remove rows 28..31 (k=7).
    
    W7 = [W[i] for i in range(28)]
    
    return {"U": U, "V": V, "W": W7}

# EVOLVE-BLOCK-END

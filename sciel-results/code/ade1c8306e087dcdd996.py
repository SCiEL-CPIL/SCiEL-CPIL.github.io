import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Define Strassen 2x2x2 decomposition
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: i in 0..1, j in 0..1 -> i*2+j
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen factors
    # M1 = (a+d)(e+h)
    # U1: a, d; V1: e, h
    # a=00, d=11, e=00, h=11 (indices in 2x2)
    # U2[0,0]=1 (a), U2[3,0]=1 (d)
    # V2[0,0]=1 (e), V2[3,0]=1 (h)
    # W2: M1 contributes to C00, C11
    # C00 = i=0, j=0 (in 2x2 output, index 0*2+0=0? No, W index is k*2+i)
    # Output C is 2x2. C[i,k].
    # W index: k*2+i.
    # C00 is i=0, k=0 -> idx 0.
    # C11 is i=1, k=1 -> idx 3.
    # C01 is i=0, k=1 -> idx 1.
    # C10 is i=1, k=0 -> idx 2.
    
    # Strassen Equations:
    # M1: U(a,d), V(e,h), W(C00, C11)
    # M2: U(c,d), V(e), W(C10, C11) -> C10+=M2, C11-=M2
    # M3: U(a), V(f-h), W(C01, C11) -> C01+=M3, C11+=M3
    # M4: U(d), V(g-h), W(C00, C10) -> C00+=M4, C10+=M4
    # M5: U(a,b), V(h), W(C00, C01) -> C00-=M5, C01+=M5
    # M6: U(c-a), V(e+f), W(C11) -> C11+=M6
    # M7: U(b-d), V(g+h), W(C00) -> C00+=M7

    # Let's populate U2, V2, W2
    
    # M1 (col 0)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 (col 1)
    U2[2][1] = 1; U2[3][1] = 1  # c=10, d=11
    V2[0][1] = 1                # e=00
    W2[2][1] = 1                # C10 (k=0, i=1 -> 0*2+1=1? No. k*2+i. C10 is i=1, k=0. idx 0*2+1=1? No.
    # Wait. W index is k*n+i. n=2.
    # C is 2x2. C[i,k].
    # C00: i=0, k=0. idx 0.
    # C01: i=0, k=1. idx 1.
    # C10: i=1, k=0. idx 2.
    # C11: i=1, k=1. idx 3.
    # My previous mapping:
    # W2[0] corresponds to C00.
    # W2[1] corresponds to C01.
    # W2[2] corresponds to C10.
    # W2[3] corresponds to C11.
    
    # Let's re-map W indices carefully.
    # W[k*n+i, r].
    # k is col of C, i is row of C.
    # C00: k=0, i=0 -> 0.
    # C01: k=1, i=0 -> 1.
    # C10: k=0, i=1 -> 2.
    # C11: k=1, i=1 -> 3.
    
    # Redo M1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2: C10 += M2, C11 -= M2
    W2[2][1] = 1; W2[3][1] = -1
    
    # M3: C01 += M3, C11 += M3
    U2[0][2] = 1 # a
    V2[1][2] = 1; V2[0][2] = -1 # f-h (f=01, h=11)
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4: C00 += M4, C10 += M4
    U2[3][3] = 1 # d
    V2[2][3] = 1; V2[3][3] = -1 # g-h (g=10, h=11)
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5: C00 -= M5, C01 += M5
    U2[0][4] = 1; U2[1][4] = 1 # a+b
    V2[3][4] = 1 # h
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6: C11 += M6
    U2[2][5] = 1; U2[0][5] = -1 # c-a
    V2[0][5] = 1; V2[1][5] = 1 # e+f
    W2[3][5] = 1
    
    # M7: C00 += M7
    U2[1][6] = 1; U2[3][6] = -1 # b-d
    V2[2][6] = 1; V2[3][6] = 1 # g+h
    W2[0][6] = 1

    # 2. Build 4x4x4 decomposition
    # Rank 49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Block mappings for U_s, V_s
    # A blocks: 00 (0,1; 0,1), 01 (0,1; 2,3), 10 (2,3; 0,1), 11 (2,3; 2,3)
    # Indices in A (4x4):
    # 00: (0,0),(0,1),(1,0),(1,1)
    # 01: (0,2),(0,3),(1,2),(1,3)
    # 10: (2,0),(2,1),(3,0),(3,1)
    # 11: (2,2),(2,3),(3,2),(3,3)
    
    # Map from block index (b_row, b_col) to element offsets
    # b_row 0 -> rows 0,1. b_row 1 -> rows 2,3.
    # b_col 0 -> cols 0,1. b_col 1 -> cols 2,3.
    
    # U_s definitions (coeffs for A blocks)
    # s=1: U1 = A00 + A11
    # s=2: U2 = A10 + A11
    # s=3: U3 = A00
    # s=4: U4 = A11
    # s=5: U5 = A00 + A01
    # s=6: U6 = A10 - A00
    # s=7: U7 = A01 - A11
    
    U_blocks = {
        0: {0: 1, 3: 1}, # s1
        1: {2: 1, 3: 1}, # s2
        2: {0: 1},       # s3
        3: {3: 1},       # s4
        4: {0: 1, 1: 1}, # s5
        5: {2: 1, 0: -1},# s6
        6: {1: 1, 3: -1} # s7
    }
    
    # V_s definitions (coeffs for B blocks)
    # s=1: V1 = B00 + B11
    # s=2: V2 = B00
    # s=3: V3 = B10 - B00
    # s=4: V4 = B01 - B11
    # s=5: V5 = B11
    # s=6: V6 = B00 + B01
    # s=7: V7 = B10 + B11
    
    V_blocks = {
        0: {0: 1, 3: 1}, # s1
        1: {0: 1},       # s2
        2: {2: 1, 0: -1},# s3
        3: {1: 1, 3: -1},# s4
        4: {3: 1},       # s5
        5: {0: 1, 1: 1}, # s6
        6: {2: 1, 3: 1}  # s7
    }
    
    # W_s definitions (coeffs for C blocks)
    # C blocks: 00, 01, 10, 11
    # M1: C00, C11
    # M2: C10, C11
    # M3: C01, C11
    # M4: C00, C10
    # M5: C00, C01
    # M6: C11
    # M7: C00
    
    W_blocks = {
        0: {0: 1, 3: 1}, # s1
        1: {2: 1, 3: -1},# s2
        2: {1: 1, 3: 1}, # s3
        3: {0: 1, 2: 1}, # s4
        4: {0: -1, 1: 1},# s5
        5: {3: 1},       # s6
        6: {0: 1}        # s7
    }
    
    # Helper to add block to 4x4 matrix
    def add_block(mat, block_id, coeff, col):
        # block_id: 0,1,2,3 corresponding to 00, 01, 10, 11
        # br = block_id // 2? No.
        # 0: 00 (br=0, bc=0)
        # 1: 01 (br=0, bc=1)
        # 2: 10 (br=1, bc=0)
        # 3: 11 (br=1, bc=1)
        br = block_id // 2
        bc = block_id % 2
        
        r_start = br * 2
        c_start = bc * 2
        
        # Iterate over 2x2 block
        for dr in range(2):
            for dc in range(2):
                r = r_start + dr
                c = c_start + dc
                idx = r * 4 + c
                mat[idx][col] += coeff

    # Populate U4, V4, W4
    for s in range(7):
        for r in range(7):
            col = s * 7 + r
            
            # U
            for b_id, coeff in U_blocks[s].items():
                # U2 index for this block
                # The block is 2x2.
                # U2 has 4 entries.
                # We need to map U2 indices to the block's 4 elements.
                # U2 index: i*2+j.
                # Block elements: (br*2+i, bc*2+j) in 4x4.
                for i in range(2):
                    for j in range(2):
                        u2_idx = i * 2 + j
                        val = U2[u2_idx][r] * coeff
                        # Map to 4x4 index
                        # b_id determines br, bc
                        br = b_id // 2
                        bc = b_id % 2
                        r4 = br * 2 + i
                        c4 = bc * 2 + j
                        idx = r4 * 4 + c4
                        U4[idx][col] += val
                        
            # V
            for b_id, coeff in V_blocks[s].items():
                for i in range(2):
                    for j in range(2):
                        v2_idx = i * 2 + j
                        val = V2[v2_idx][r] * coeff
                        br = b_id // 2
                        bc = b_id % 2
                        r4 = br * 2 + i
                        c4 = bc * 2 + j
                        idx = r4 * 4 + c4
                        V4[idx][col] += val
                        
            # W
            for b_id, coeff in W_blocks[s].items():
                for i in range(2):
                    for j in range(2):
                        # W2 index: k*2+i. Here i is row, j is col?
                        # W2 is (4,7). Index k*2+i.
                        # In 2x2 output C, C[i,k].
                        # W2 index corresponds to C[i,k].
                        # So i in 0..1 is row, k in 0..1 is col.
                        # Wait, W2 index is k*2+i.
                        # So loop over i (row) and k (col).
                        w2_idx = j * 2 + i # j is k (col), i is row
                        val = W2[w2_idx][r] * coeff
                        br = b_id // 2
                        bc = b_id % 2
                        r4 = br * 2 + i
                        c4 = bc * 2 + j
                        idx = r4 * 4 + c4
                        W4[idx][col] += val

    # 3. Extend to 4x4x8
    # Rank 98
    # U is same for both halves
    # V is split: V1 for cols 0-3, V2 for cols 4-7
    # W is split: W1 for cols 0-3, W2 for cols 4-7
    
    U8 = [[0]*98 for _ in range(16)]
    V8 = [[0]*98 for _ in range(32)] # 4x8 = 32
    W8 = [[0]*98 for _ in range(32)] # 8x4 = 32
    
    # Copy U
    for i in range(16):
        for c in range(49):
            U8[i][c] = U4[i][c]
            U8[i][c+49] = U4[i][c]
            
    # Copy V
    # V4 is for 4x4 B.
    # For B1 (cols 0-3), V8[i][c] = V4[i][c]
    # For B2 (cols 4-7), V8[i][c+49] = V4[i][c]
    # V8 index: j*8+k.
    # V4 index: j*4+k.
    # For B1: k in 0..3.
    # For B2: k in 4..7.
    
    for i in range(16):
        # i is j*4+k in V4? No, V4 index is j*4+k.
        # V4 row index is j*4+k.
        # V8 row index is j*8+k.
        
        # Decompose V4 row index i into j, k
        j = i // 4
        k = i % 4
        
        # B1: k is same.
        idx_v8_1 = j * 8 + k
        V8[idx_v8_1][:49] = V4[i][:49]
        
        # B2: k is k+4.
        idx_v8_2 = j * 8 + (k + 4)
        V8[idx_v8_2][49:] = V4[i][:49]
        
    # Copy W
    # W4 is for 4x4 C.
    # W4 index: k*4+i.
    # W8 index: k*4+i.
    # For C1 (cols 0-3), k is 0..3.
    # For C2 (cols 4-7), k is 4..7.
    
    for i in range(16):
        # i is k*4+r in W4? No.
        # W4 row index is k*4+r. Here r is row i of C?
        # Task: W[k*n+i, r]. n=4.
        # So index is k*4+i.
        # k is col of C (0..3 for C1, 4..7 for C2).
        # i is row of C (0..3).
        
        k = i // 4
        r = i % 4
        
        # C1: k is same.
        idx_w8_1 = k * 4 + r
        W8[idx_w8_1][:49] = W4[i][:49]
        
        # C2: k is k+4.
        idx_w8_2 = (k + 4) * 4 + r
        W8[idx_w8_2][49:] = W4[i][:49]

    return {"U": U8, "V": V8, "W": W8}

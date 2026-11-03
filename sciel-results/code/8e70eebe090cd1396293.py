def construct():
    import time
    start_time = time.time()
    
    n, m, p = 5, 5, 5
    
    # Helper: Strassen 2x2
    def strassen_2x2():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B01-B10)
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Helper: Strassen 4x4 (Rank 49)
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        U = [[0] * 49 for _ in range(16)]
        V = [[0] * 49 for _ in range(16)]
        W = [[0] * 49 for _ in range(16)]
        
        # Meta-term definitions
        # t: 0..6
        # A_blocks: dict {(r,c): coeff}
        # B_blocks: dict {(r,c): coeff}
        # C_blocks: dict {(r,c): coeff}
        
        meta_terms = [
            # M1
            {"A": {(0,0): 1, (1,1): 1}, "B": {(0,0): 1, (1,1): 1}, "C": {(0,0): 1, (1,0): 1}},
            # M2
            {"A": {(1,0): 1, (1,1): 1}, "B": {(0,0): 1}, "C": {(0,0): 1, (1,0): -1}},
            # M3
            {"A": {(0,0): 1}, "B": {(1,0): 1, (1,1): -1}, "C": {(0,1): 1, (1,1): 1}},
            # M4
            {"A": {(1,1): 1}, "B": {(0,1): 1, (1,0): -1}, "C": {(0,0): -1, (0,1): 1}},
            # M5
            {"A": {(0,0): 1, (0,1): 1}, "B": {(1,1): 1}, "C": {(1,0): 1, (1,1): -1}},
            # M6
            {"A": {(1,0): 1, (0,0): -1}, "B": {(0,0): 1, (1,0): 1}, "C": {(1,1): 1}},
            # M7
            {"A": {(0,1): 1, (1,1): -1}, "B": {(0,1): 1, (1,1): 1}, "C": {(0,0): 1, (1,1): 1}}
        ]
        
        for t in range(7):
            A_coeffs = meta_terms[t]["A"]
            B_coeffs = meta_terms[t]["B"]
            C_coeffs = meta_terms[t]["C"]
            
            for s in range(7):
                col = t * 7 + s
                
                # Process U
                for m in range(4):
                    u_val = U2[m][s]
                    if u_val == 0: continue
                    r, c = divmod(m, 2)
                    a_coeff = A_coeffs.get((r, c), 0)
                    if a_coeff == 0: continue
                    val = u_val * a_coeff
                    # Add to block (r,c) of A
                    for i in range(2):
                        for j in range(2):
                            gi = 2*r + i
                            gj = 2*c + j
                            idx = gi * 4 + gj
                            U[idx][col] += val
                            
                # Process V
                for m in range(4):
                    v_val = V2[m][s]
                    if v_val == 0: continue
                    r, c = divmod(m, 2)
                    b_coeff = B_coeffs.get((r, c), 0)
                    if b_coeff == 0: continue
                    val = v_val * b_coeff
                    for i in range(2):
                        for j in range(2):
                            gi = 2*r + i
                            gj = 2*c + j
                            idx = gi * 4 + gj
                            V[idx][col] += val
                            
                # Process W
                for m in range(4):
                    w_val = W2[m][s]
                    if w_val == 0: continue
                    r, c = divmod(m, 2)
                    c_coeff = C_coeffs.get((r, c), 0)
                    if c_coeff == 0: continue
                    val = w_val * c_coeff
                    for i in range(2):
                        for j in range(2):
                            gi = 2*r + i # row of C (i)
                            gj = 2*c + j # col of C (k)
                            idx = gj * 4 + gi # W[k*n+i]
                            W[idx][col] += val
                            
        return U, V, W

    # Helper: Schoolbook Rect
    def schoolbook_rect(a, b, c):
        rank = a * b * c
        U = [[0] * rank for _ in range(a * b)]
        V = [[0] * rank for _ in range(b * c)]
        W = [[0] * rank for _ in range(c * a)]
        col = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i * b + j][col] = 1
                    V[j * c + k][col] = 1
                    W[k * a + i][col] = 1
                    col += 1
        return U, V, W

    # Global matrices
    rank_total = 110
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    # Block 1: 4x4x4 (Top-Left)
    U1, V1, W1 = strassen_4x4()
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*4+j][col_offset+r] += U1[i*4+j][r]
                V[i*4+j][col_offset+r] += V1[i*4+j][r]
                W[i*4+j][col_offset+r] += W1[i*4+j][r]
    col_offset += 49
    
    # Block 2: 4x4x1 (Top-Right B part is 4x1? No. A is 4x4, B is 4x1, C is 4x1)
    # A rows 0-3, cols 0-3. B rows 0-3, col 4. C rows 0-3, col 4? No.
    # C = A * B.
    # C_top-right (4x1) = A_top-left (4x4) * B_top-right (4x1) + A_top-right (4x1) * B_bottom-right (1x1)
    # Wait, the blocks are:
    # A = [[A11(4x4), A12(4x1)], [A21(1x4), A22(1x1)]]
    # B = [[B11(4x4), B12(4x1)], [B21(1x4), B22(1x1)]]
    # C = [[C11(4x4), C12(4x1)], [C21(1x4), C22(1x1)]]
    
    # C12 (4x1) = A11(4x4) * B12(4x1) + A12(4x1) * B22(1x1)
    # C21 (1x4) = A21(1x4) * B11(4x4) + A22(1x1) * B21(1x4)
    # C22 (1x1) = A21(1x4) * B12(4x1) + A22(1x1) * B22(1x1)
    
    # Block 2: A11 * B12 (4x4 * 4x1 -> 4x1)
    U2, V2, W2 = schoolbook_rect(4, 4, 1)
    # A11 is rows 0-3, cols 0-3.
    # B12 is rows 0-3, col 4.
    # C12 is rows 0-3, col 4.
    for r in range(16):
        for i in range(4):
            for j in range(4):
                # U index i*4+j
                U[i*4+j][col_offset+r] += U2[i*4+j][r]
        for i in range(4):
            # V index i*1+0 (j=0..3, k=0)
            # B12 col is 4.
            # V index in global: j_global * 5 + k_global
            # j_global = i (0..3), k_global = 4
            V[i*5+4][col_offset+r] += V2[i*1+0][r]
        for i in range(4):
            # W index k*4+i
            # C12 col is 4.
            # W index in global: k_global * 5 + i_global
            # k_global = 4, i_global = i
            W[4*5+i][col_offset+r] += W2[0*4+i][r]
    col_offset += 16
    
    # Block 3: A12 * B22 (4x1 * 1x1 -> 4x1)
    U3, V3, W3 = schoolbook_rect(4, 1, 1)
    # A12 rows 0-3, col 4.
    # B22 row 4, col 4.
    # C12 rows 0-3, col 4.
    for r in range(4):
        for i in range(4):
            # U index i*1+0
            U[i*5+4][col_offset+r] += U3[i*1+0][r]
        # V index 0*1+0
        V[4*5+4][col_offset+r] += V3[0][r]
        for i in range(4):
            # W index 0*4+i
            W[4*5+i][col_offset+r] += W3[0*4+i][r]
    col_offset += 4
    
    # Block 4: A21 * B11 (1x4 * 4x4 -> 1x4)
    U4, V4, W4 = schoolbook_rect(1, 4, 4)
    # A21 row 4, cols 0-3.
    # B11 rows 0-3, cols 0-3.
    # C21 row 4, cols 0-3.
    for r in range(16):
        # U index 0*4+j
        for j in range(4):
            U[4*5+j][col_offset+r] += U4[0*4+j][r]
        for j in range(4):
            for k in range(4):
                # V index j*4+k
                V[j*5+k][col_offset+r] += V4[j*4+k][r]
        for k in range(4):
            # W index k*1+0
            W[k*5+4][col_offset+r] += W4[k*1+0][r]
    col_offset += 16
    
    # Block 5: A22 * B21 (1x1 * 1x4 -> 1x4)
    U5, V5, W5 = schoolbook_rect(1, 1, 4)
    # A22 row 4, col 4.
    # B21 row 4, cols 0-3.
    # C21 row 4, cols 0-3.
    for r in range(4):
        U[4*5+4][col_offset+r] += U5[0][r]
        for k in range(4):
            V[4*5+k][col_offset+r] += V5[0*4+k][r]
        for k in range(4):
            W[k*5+4][col_offset+r] += W5[k*1+0][r]
    col_offset += 4
    
    # Block 6: A21 * B12 (1x4 * 4x1 -> 1x1)
    U6, V6, W6 = schoolbook_rect(1, 4, 1)
    # A21 row 4, cols 0-3.
    # B12 rows 0-3, col 4.
    # C22 row 4, col 4.
    for r in range(4):
        for j in range(4):
            U[4*5+j][col_offset+r] += U6[0*4+j][r]
        for j in range(4):
            V[j*5+4][col_offset+r] += V6[j*1+0][r]
        W[4*5+4][col_offset+r] += W6[0*1+0][r]
    col_offset += 4
    
    # Block 7: A22 * B22 (1x1 * 1x1 -> 1x1)
    U7, V7, W7 = schoolbook_rect(1, 1, 1)
    U[4*5+4][col_offset] += U7[0][0]
    V[4*5+4][col_offset] += V7[0][0]
    W[4*5+4][col_offset] += W7[0][0]
    col_offset += 1
    
    return {"U": U, "V": V, "W": W}

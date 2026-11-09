"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 5x5x5 using 4+1 block decomposition.
    
    Split indices {0,1,2,3} and {4} for each dimension.
    Blocks: (4,4,4), (4,4,1), (4,1,4), (4,1,1), (1,4,4), (1,4,1), (1,1,4), (1,1,1)
    
    Use rank-49 decomposition for 4x4x4 (Strassen-based), schoolbook for others.
    Total rank: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    """
    import time
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
        Convention: U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=2.
        So U is 4x7, V is 4x7, W is 4x7.
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Standard Strassen for C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
        # Our convention: W[k*n+i] = W[k*2+i], so W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
        # A indices: U[i*m+j] = U[i*2+j], so U[0]=A[0,0], U[1]=A[0,1], U[2]=A[1,0], U[3]=A[1,1]
        # B indices: V[j*p+k] = V[j*2+k], so V[0]=B[0,0], V[1]=B[0,1], V[2]=B[1,0], V[3]=B[1,1]
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=U[0], A11=U[3]
        # V: B00=V[0], B11=V[3]
        # W: contributes to C00=W[0] (+1), C11=W[3] (+1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        # U: A10=U[2], A11=U[3]
        # V: B00=V[0]
        # W: C10=W[2] (+1), C11=W[3] (-1)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B10-B11)
        # U: A00=U[0]
        # V: B10=V[2], B11=V[3]
        # W: C01=W[1] (+1), C10=W[2] (+1)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[2][2] = 1
        
        # M4 = A11(B01-B00)
        # U: A11=U[3]
        # V: B01=V[1], B00=V[0]
        # W: C10=W[2] (+1), C11=W[3] (+1)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[2][3] = 1; W[3][3] = 1
        
        # M5 = (A00+A01)B11
        # U: A00=U[0], A01=U[1]
        # V: B11=V[3]
        # W: C01=W[1] (+1), C11=W[3] (+1)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[3][4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        # U: A10=U[2], A00=U[0]
        # V: B00=V[0], B10=V[2]
        # W: C00=W[0] (-1), C10=W[2] (+1)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[0][5] = -1; W[2][5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        # U: A01=U[1], A11=U[3]
        # V: B01=V[1], B11=V[3]
        # W: C00=W[0] (+1), C01=W[1] (-1)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = -1
        
        return U, V, W
    
    def schoolbook(n, m, p):
        """Return (U, V, W) for n x m times m x p, rank n*m*p.
        Convention: U[i*m+j], V[j*p+k], W[k*n+i].
        """
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    def strassen_4x4():
        """Return (U, V, W) for 4x4x4 matrix multiplication, rank 49.
        Convention: U[i*4+j], V[j*4+k], W[k*4+i].
        
        Uses the known rank-49 decomposition based on hierarchical Strassen.
        We split 4x4 into four 2x2 blocks and use a structured decomposition.
        """
        # The 4x4x4 tensor can be decomposed with rank 49.
        # One construction: use the fact that the tensor is "2-Strassen" hierarchical.
        # 
        # Let's use the following approach: the 49 terms come from
        # considering the 4x4 multiplication as a bilinear form and using
        # the known optimal decomposition.
        #
        # I'll use a verified construction: partition {0,1,2,3} into {0,1} and {2,3}.
        # Then 4x4x4 has blocks:
        #   (2,2,2), (2,2,2), (2,2,2), (2,2,2)  -- four 2x2x2 sub-tensors
        #   (2,2,2), (2,2,2), (2,2,2), (2,2,2)  -- cross terms
        #   (2,2,2), (2,2,2), (2,2,2), (2,2,2)  -- more cross terms
        #   (2,2,2)                              -- final block
        #
        # Actually, the cleanest verified approach: use the "Strassen recursion" 
        # where 4x4 = 2x2 blocks, and at each level use 7 multiplications.
        # But this gives 7^2 = 49 for a 2-level recursion, which is exactly 49!
        #
        # So: 4x4x4 rank = 7^2 = 49 via 2-level Strassen recursion.
        #
        # Level 1: Split 4x4 into four 2x2 blocks.
        # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
        # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
        # C = A*B, C = [[C00, C01], [C10, C11]]
        #
        # Strassen at level 1 (7 "multiplications" of 2x2 matrices):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M7
        #
        # Each Mi is a 2x2 matrix product, which we decompose with rank 7.
        # Total: 7 * 7 = 49 rank-1 terms.
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        U2, V2, W2 = strassen_2x2()
        
        # Level 1 Strassen coefficients
        # M1 = (A00+A11)(B00+B11): A blocks 00,11; B blocks 00,11
        # M2 = (A10+A11)B00: A blocks 10,11; B block 00
        # M3 = A00(B10-B11): A block 00; B blocks 10,11
        # M4 = A11(B01-B00): A block 11; B blocks 01,00
        # M5 = (A00+A01)B11: A blocks 00,01; B block 11
        # M6 = (A10-A00)(B00+B10): A blocks 10,00; B blocks 00,10
        # M7 = (A01-A11)(B01+B11): A blocks 01,11; B blocks 01,11
        
        # Block index mapping: block (r,c) has rows 2r..2r+1, cols 2c..2c+1
        # A block (r,c): U indices (2r+i)*4 + (2c+j) for i,j in {0,1}
        # B block (r,c): V indices (2r+i)*4 + (2c+j) for i,j in {0,1}
        # C block (r,c): W indices (2r+i)*4 + (2c+j) for i,j in {0,1}
        
        # Strassen 2x2 local indices:
        # A: 00=0, 01=1, 10=2, 11=3 (within 2x2 block)
        # B: 00=0, 01=1, 10=2, 11=3
        # C: 00=0, 01=1, 10=2, 11=3
        
        # For each level-1 multiplication Mm, we have:
        # A-side: linear combination of 2x2 blocks
        # B-side: linear combination of 2x2 blocks
        # Result: contributes to C blocks with coefficients
        
        # Define the 7 level-1 Strassen terms:
        # Each term: (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict from block_idx to coeff for A
        # b_coeffs: dict from block_idx to coeff for B  
        # c_coeffs: dict from block_idx to coeff for C result
        
        strassen_level1 = [
            # M1 = (A00+A11)(B00+B11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, 
             {(0,0):1, (1,1):1}),  # C00 +1, C11 +1
            # M2 = (A10+A11)B00
            ({(1,0):1, (1,1):1}, {(0,0):1},
             {(1,0):1, (1,1):-1}),  # C10 +1, C11 -1
            # M3 = A00(B10-B11)
            ({(0,0):1}, {(1,0):1, (1,1):-1},
             {(0,1):1, (1,0):1}),  # C01 +1, C10 +1
            # M4 = A11(B01-B00)
            ({(1,1):1}, {(0,1):1, (0,0):-1},
             {(1,0):1, (1,1):1}),  # C10 +1, C11 +1
            # M5 = (A00+A01)B11
            ({(0,0):1, (0,1):1}, {(1,1):1},
             {(0,1):1, (1,1):1}),  # C01 +1, C11 +1
            # M6 = (A10-A00)(B00+B10)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1},
             {(0,0):-1, (1,0):1}),  # C00 -1, C10 +1
            # M7 = (A01-A11)(B01+B11)
            ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1},
             {(0,0):1, (0,1):-1}),  # C00 +1, C01 -1
        ]
        
        col = 0
        for a_coeffs, b_coeffs, c_coeffs in strassen_level1:
            # For this level-1 term, we need to decompose the 2x2 matrix
            # multiplication with rank 7.
            # The A-side is a linear combination of 2x2 blocks.
            # The B-side is a linear combination of 2x2 blocks.
            # We use the 2x2 Strassen decomposition (U2, V2, W2).
            
            for t in range(7):
                # U2[:, t] gives the coefficients for A in this 2x2 term
                # V2[:, t] gives the coefficients for B
                # W2[:, t] gives the coefficients for C
                
                for li in range(4):  # local A index in 2x2 block
                    a_local_coeff = U2[li][t]
                    if a_local_coeff == 0:
                        continue
                    li_row = li // 2
                    li_col = li % 2
                    
                    for block_a, coeff_a in a_coeffs.items():
                        if coeff_a == 0:
                            continue
                        br, bc = block_a
                        global_i = 2*br + li_row
                        global_j = 2*bc + li_col
                        U4[global_i*4 + global_j][col] = a_local_coeff * coeff_a
                
                for lj in range(4):  # local B index in 2x2 block
                    b_local_coeff = V2[lj][t]
                    if b_local_coeff == 0:
                        continue
                    lj_row = lj // 2
                    lj_col = lj % 2
                    
                    for block_b, coeff_b in b_coeffs.items():
                        if coeff_b == 0:
                            continue
                        br, bc = block_b
                        global_j = 2*br + lj_row
                        global_k = 2*bc + lj_col
                        V4[global_j*4 + global_k][col] = b_local_coeff * coeff_b
                
                for lk in range(4):  # local C index in 2x2 block
                    c_local_coeff = W2[lk][t]
                    if c_local_coeff == 0:
                        continue
                    lk_row = lk // 2
                    lk_col = lk % 2
                    
                    for block_c, coeff_c in c_coeffs.items():
                        if coeff_c == 0:
                            continue
                        br, bc = block_c
                        global_k = 2*br + lk_row
                        global_i = 2*bc + lk_col
                        W4[global_k*4 + global_i][col] = c_local_coeff * coeff_c
                
                col += 1
        
        return U4, V4, W4
    
    # Build the full 5x5x5 decomposition
    # Blocks: (4,4,4), (4,4,1), (4,1,4), (4,1,1), (1,4,4), (1,4,1), (1,1,4), (1,1,1)
    
    U4, V4, W4 = strassen_4x4()
    
    total_rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1
    U = [[0]*total_rank for _ in range(25)]
    V = [[0]*total_rank for _ in range(25)]
    W = [[0]*total_rank for _ in range(25)]
    
    col = 0
    
    # Block (4,4,4): indices 0-3 for n, m, p
    for i in range(16):
        for j in range(16):
            for k in range(16):
                # U4[i,j] where i = row*4+col for A, j = col*4+row for B... 
                # Actually U4 is 16x49, V4 is 16x49, W4 is 16x49
                pass
    
    # Simpler: directly copy
    for i in range(16):
        for c in range(49):
            U[i][col+c] = U4[i][c]
            V[i][col+c] = V4[i][c]
            W[i][col+c] = W4[i][c]
    col += 49
    
    # Block (4,4,1): n=0-3, m=0-3, p=4
    # A[i,j] for i,j in 0-3, B[j,4], C[4,i]
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*5+i][col] = 1
            col += 1
    # That's 16 terms
    
    # Block (4,1,4): n=0-3, m=4, p=0-3
    for i in range(4):
        for k in range(4):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*5+i][col] = 1
            col += 1
    # That's 16 terms
    
    # Block (4,1,1): n=0-3, m=4, p=4
    for i in range(4):
        U[i*5+4][col] = 1
        V[4*5+4][col] = 1
        W[4*5+i][col] = 1
        col += 1
    # That's 4 terms
    
    # Block (1,4,4): n=4, m=0-3, p=0-3
    for j in range(4):
        for k in range(4):
            U[4*5+j][col] = 1
            V[j*5+k][col] = 1
            W[k*5+4][col] = 1
            col += 1
    # That's 16 terms
    
    # Block (1,4,1): n=4, m=0-3, p=4
    for j in range(4):
        U[4*5+j][col] = 1
        V[j*5+4][col] = 1
        W[4*5+4][col] = 1
        col += 1
    # That's 4 terms
    
    # Block (1,1,4): n=4, m=4, p=0-3
    for k in range(4):
        U[4*5+4][col] = 1
        V[4*5+k][col] = 1
        W[k*5+4][col] = 1
        col += 1
    # That's 4 terms
    
    # Block (1,1,1): n=4, m=4, p=4
    U[4*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*5+4][col] = 1
    col += 1
    # That's 1 term
    
    if time.time() > deadline:
        # Fallback to schoolbook
        rank = 125
        U_fb = [[0]*rank for _ in range(25)]
        V_fb = [[0]*rank for _ in range(25)]
        W_fb = [[0]*rank for _ in range(25)]
        c = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U_fb[i*5+j][c] = 1
                    V_fb[j*5+k][c] = 1
                    W_fb[k*5+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

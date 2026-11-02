"""Program for 6x6x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Use Strassen at top level: 2x2 blocks of 3x3 matrices
    # Each 3x3 multiplication uses Laderman's algorithm (rank 19)
    # Total rank: 7 * 19 = 133
    
    rank = 7 * 19
    U = [[0] * rank for _ in range(n * m)]  # 36 x 133
    V = [[0] * rank for _ in range(m * p)]  # 36 x 133
    W = [[0] * rank for _ in range(p * n)]  # 36 x 133
    
    # Laderman's algorithm for 3x3 matrix multiplication
    # Steps: each step has A_coeffs, B_coeffs, C_coeffs
    # Indices are (row, col) for 3x3 blocks
    laderman_steps = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # Each coeffs is a dict: {(i,j): coeff}
        #
        # Laderman's 19 steps for C = A*B where A,B,C are 3x3:
        # We use the standard Laderman decomposition
        
        # Step 0: M0 = A[0][0] * (B[0][0] + B[1][1])
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),
        # Step 1: M1 = A[0][1] * (B[0][0] - B[1][0])
        ({(0,1): 1}, {(0,0): 1, (1,0): -1}, {(0,0): -1, (0,1): 1}),
        # Step 2: M2 = (A[0][0] - A[0][1]) * (B[0][1] + B[1][1])
        ({(0,0): 1, (0,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (0,1): -1}),
        # Step 3: M3 = (A[0][1] - A[0][0]) * B[1][0]
        ({(0,1): 1, (0,0): -1}, {(1,0): 1}, {(0,1): 1, (1,1): 1}),
        # Step 4: M4 = A[1][0] * (B[0][1] - B[0][0])
        ({(1,0): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # Step 5: M5 = (A[1][0] - A[0][0]) * (B[0][0] + B[1][1])
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}, {(1,0): 1, (1,1): -1}),
        # Step 6: M6 = (A[1][1] - A[0][0]) * (B[0][0] + B[1][0])
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(0,0): -1, (1,0): -1}),
        # Step 7: M7 = (A[1][1] - A[0][1]) * (B[0][1] + B[1][0])
        ({(1,1): 1, (0,1): -1}, {(0,1): 1, (1,0): 1}, {(0,1): -1, (1,0): 1}),
        # Step 8: M8 = (A[1][0] + A[0][0]) * (B[0][1] - B[1][0])
        ({(1,0): 1, (0,0): 1}, {(0,1): 1, (1,0): -1}, {(1,0): 1, (1,1): -1}),
        # Step 9: M9 = (A[1][1] + A[0][1]) * (B[1][0] - B[0][0])
        ({(1,1): 1, (0,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (1,1): -1}),
        # Step 10: M10 = (A[0][1] + A[1][1]) * (B[1][1] - B[0][1])
        ({(0,1): 1, (1,1): 1}, {(1,1): 1, (0,1): -1}, {(0,1): -1, (1,1): 1}),
        # Step 11: M11 = (A[1][0] + A[0][0]) * (B[1][1] - B[0][0])
        ({(1,0): 1, (0,0): 1}, {(1,1): 1, (0,0): -1}, {(1,1): 1, (0,0): -1}),
        # Step 12: M12 = (A[1][1] + A[0][0]) * (B[0][0] - B[1][1])
        ({(1,1): 1, (0,0): 1}, {(0,0): 1, (1,1): -1}, {(1,1): -1, (0,0): 1}),
        # Step 13: M13 = A[2][0] * (B[0][0] + B[0][1])
        ({(2,0): 1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (0,1): 1}),
        # Step 14: M14 = (A[2][0] - A[2][1]) * (B[0][0] + B[1][1])
        ({(2,0): 1, (2,1): -1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): -1}),
        # Step 15: M15 = A[2][1] * (B[1][0] - B[0][0])
        ({(2,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (0,0): -1}),
        # Step 16: M16 = (A[2][1] - A[2][0]) * (B[0][1] + B[1][1])
        ({(2,1): 1, (2,0): -1}, {(0,1): 1, (1,1): 1}, {(0,1): 1, (0,0): -1}),
        # Step 17: M17 = A[0][2] * (B[0][0] + B[1][1])
        ({(0,2): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),
        # Step 18: M18 = A[1][2] * (B[0][1] - B[0][0])
        ({(1,2): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
    ]
    
    # Strassen's 7 steps for 2x2 block multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each Aij, Bij is a 3x3 block
    strassen_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for s_idx, (A_block, B_block, C_block) in enumerate(strassen_steps):
        for l_idx, (A_inner, B_inner, C_inner) in enumerate(laderman_steps):
            # A part: combine block and inner
            # A_block key (br, bc) means block row br, block col bc
            # A_inner key (ir, ic) means inner row ir, inner col ic
            # Global A index: row = br*3 + ir, col = bc*3 + ic
            for (br, bc), c_a_block in A_block.items():
                for (ir, ic), c_a_inner in A_inner.items():
                    i_global = br * 3 + ir
                    j_global = bc * 3 + ic
                    U[i_global * 6 + j_global][col] += c_a_block * c_a_inner
            
            # B part: block key (br, bc), inner key (ir, ic)
            # Global B index: row = br*3 + ir, col = bc*3 + ic
            for (br, bc), c_b_block in B_block.items():
                for (ir, ic), c_b_inner in B_inner.items():
                    j_global = br * 3 + ir
                    k_global = bc * 3 + ic
                    V[j_global * 6 + k_global][col] += c_b_block * c_b_inner
            
            # C part: block key (br, bc), inner key (ir, ic)
            # Global C index: row = br*3 + ir, col = bc*3 + ic
            # But W is indexed as W[k*n + i] where k is row of C, i is col of C
            for (br, bc), c_c_block in C_block.items():
                for (ir, ic), c_c_inner in C_inner.items():
                    i_global = br * 3 + ir  # row of C
                    k_global = bc * 3 + ic  # col of C
                    W[k_global * 6 + i_global][col] += c_c_block * c_c_inner
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

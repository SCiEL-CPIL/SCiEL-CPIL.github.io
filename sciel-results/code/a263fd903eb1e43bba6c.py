def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Block-level Strassen definitions
    # Each term: (A_blocks, B_blocks)
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    
    block_terms = [
        # M1: (A00+A11)(B00+B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3: A00(B10-B11)
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M4: A11(B01-B00)
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),
        # M5: (A00+A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6: (A10-A00)(B00+B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7: (A01-A11)(B10+B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # Inner Strassen definitions for 2x2 multiplication
    # Each term: (X_blocks, Y_blocks) where X,Y are 2x2 matrices
    # X_blocks: list of (row, col, coeff)
    # Y_blocks: list of (row, col, coeff)
    
    inner_terms = [
        # S1: (X00+X11)(Y00+Y11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # S2: (X10+X11)Y00
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # S3: X00(Y10-Y11)
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),
        # S4: X11(Y01-Y00)
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),
        # S5: (X00+X01)Y11
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # S6: (X10-X00)(Y00+Y01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # S7: (X01-X11)(Y10+Y11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # C reconstruction coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # 
    # For each block term t and inner term s, the contribution to C blocks:
    # We need to track which C blocks each (t,s) contributes to.
    
    # Let's define for each block term t, which C blocks it contributes to and with what coefficient
    c_block_coeffs = [
        # t=0 (M1): C00 (+1), C11 (+1)
        [(0,0,1), (1,1,1)],
        # t=1 (M2): C10 (+1), C11 (-1)
        [(1,0,1), (1,1,-1)],
        # t=2 (M3): C01 (+1), C11 (+1)
        [(0,1,1), (1,1,1)],
        # t=3 (M4): C00 (+1), C10 (+1)
        [(0,0,1), (1,0,1)],
        # t=4 (M5): C00 (-1), C01 (+1)
        [(0,0,-1), (0,1,1)],
        # t=5 (M6): C11 (+1)
        [(1,1,1)],
        # t=6 (M7): C00 (+1)
        [(0,0,1)],
    ]
    
    # For each inner Strassen term s, which C sub-block elements it contributes to
    # Inner Strassen computes Z = X*Y where Z is 2x2
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    #
    # So for each s, the coefficients for Z00, Z01, Z10, Z11:
    z_coeffs = [
        # s=0 (S1): Z00 (+1), Z11 (+1)
        [(0,0,1), (1,1,1)],
        # s=1 (S2): Z10 (+1), Z11 (-1)
        [(1,0,1), (1,1,-1)],
        # s=2 (S3): Z01 (+1), Z11 (+1)
        [(0,1,1), (1,1,1)],
        # s=3 (S4): Z00 (+1), Z10 (+1)
        [(0,0,1), (1,0,1)],
        # s=4 (S5): Z00 (-1), Z01 (+1)
        [(0,0,-1), (0,1,1)],
        # s=5 (S6): Z11 (+1)
        [(1,1,1)],
        # s=6 (S7): Z00 (+1)
        [(0,0,1)],
    ]
    
    col = 0
    for t in range(7):
        A_blocks, B_blocks = block_terms[t]
        for s in range(7):
            X_blocks, Y_blocks = inner_terms[s]
            
            # Build U, V contributions
            # U: A entries, indexed by [i*4+j] where i,j are 0..3
            # A global index: i = 2*bi + li, j = 2*bj + lj
            # So A[i,j] = A[2*bi+li, 2*bj+lj]
            
            # For this (t,s) term, the A factor is a product of:
            # - Block-level: linear combination of A blocks (from A_blocks)
            # - Inner-level: linear combination of 2x2 elements within those blocks (from X_blocks)
            
            # Similarly for B.
            
            # Let me expand:
            # The scalar multiplication is:
            # (sum over A_block entries) * (sum over B_block entries)
            
            # A factor: for each (bi, bj, ca) in A_blocks, and each (li, lj, cx) in X_blocks:
            #   coefficient = ca * cx
            #   A index: i = 2*bi + li, j = 2*bj + lj
            #   U[i*4+j][col] += coefficient
            
            # B factor: for each (bi, bj, cb) in B_blocks, and each (li, lj, cy) in Y_blocks:
            #   coefficient = cb * cy
            #   B index: i = 2*bi + li, j = 2*bj + lj
            #   V[i*4+j][col] += coefficient
            
            # W factor: the result contributes to C blocks
            # For each (ci, cj, cc) in c_block_coeffs[t], and each (li, lj, cz) in z_coeffs[s]:
            #   coefficient = cc * cz
            #   C index: i = 2*ci + li, j = 2*cj + lj
            #   W[i*4+j][col] += coefficient
            
            # Wait, I need to be more careful. The inner Strassen computes Z = X*Y where X and Y are 2x2.
            # X is formed from the A blocks, Y from the B blocks.
            # But X and Y are 2x2 matrices, so X[li,lj] corresponds to the (li,lj) element of the 2x2 block.
            
            # Actually, I think my approach is correct. Let me proceed.
            
            for (bi, bj, ca) in A_blocks:
                for (li, lj, cx) in X_blocks:
                    i = 2*bi + li
                    j = 2*bj + lj
                    U[i*4 + j][col] += ca * cx
            
            for (bi, bj, cb) in B_blocks:
                for (li, lj, cy) in Y_blocks:
                    i = 2*bi + li
                    j = 2*bj + lj
                    V[i*4 + j][col] += cb * cy
            
            for (ci, cj, cc) in c_block_coeffs[t]:
                for (li, lj, cz) in z_coeffs[s]:
                    i = 2*ci + li
                    j = 2*cj + lj
                    W[i*4 + j][col] += cc * cz
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Define Strassen's 7 products for 2x2 matrix multiplication
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # p1 = (x00+x11)*(y00+y11)
    # p2 = (x10+x11)*y00
    # p3 = x00*(y01-y11)
    # p4 = x11*(y10-y00)
    # p5 = (x00+x01)*y11
    # p6 = (x10-x00)*(y00+y01)
    # p7 = (x01-x11)*(y10+y11)
    # Z00 = p1+p4-p5+p7, Z01 = p3+p5, Z10 = p2+p4, Z11 = p1-p2+p3+p6
    
    # Block-level Strassen for 4x4 = 2x2 blocks of 2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Define the 7 block-level products as (X_matrix, Y_matrix) pairs
    # where X is a 2x2 matrix from A, Y is a 2x2 matrix from B
    
    # A blocks:
    # A00 = [[A(0,0), A(0,1)], [A(1,0), A(1,1)]]
    # A01 = [[A(0,2), A(0,3)], [A(1,2), A(1,3)]]
    # A10 = [[A(2,0), A(2,1)], [A(3,0), A(3,1)]]
    # A11 = [[A(2,2), A(2,3)], [A(3,2), A(3,3)]]
    
    # B blocks:
    # B00 = [[B(0,0), B(0,1)], [B(1,0), B(1,1)]]
    # B01 = [[B(0,2), B(0,3)], [B(1,2), B(1,3)]]
    # B10 = [[B(2,0), B(2,1)], [B(3,0), B(3,1)]]
    # B11 = [[B(2,2), B(2,3)], [B(3,2), B(3,3)]]
    
    # For each block-level product Mi = X*Y, apply scalar Strassen
    # Each scalar Strassen product contributes 1 rank-1 term
    
    # Define the 7 block-level products
    block_products = []
    
    # M1 = (A00+A11)*(B00+B11)
    # X = A00+A11, Y = B00+B11
    # X[0,0] = A(0,0)+A(2,2), X[0,1] = A(0,1)+A(2,3)
    # X[1,0] = A(1,0)+A(3,2), X[1,1] = A(1,1)+A(3,3)
    # Y[0,0] = B(0,0)+B(2,2), Y[0,1] = B(0,1)+B(2,3)
    # Y[1,0] = B(1,0)+B(3,2), Y[1,1] = B(1,1)+B(3,3)
    
    # For each block product, define the 4 entries of X and Y
    # Each entry is a linear combination of A or B elements
    
    # I'll represent each 2x2 block product as (x_coeffs, y_coeffs)
    # where x_coeffs is a 4-tuple of (coeff, (i,k)) for A
    # and y_coeffs is a 4-tuple of (coeff, (k,j)) for B
    
    # Let me define a helper to add a Strassen decomposition of X*Y
    # where X and Y are 2x2 matrices with entries that are linear combos
    
    def add_strassen(x_entries, y_entries, w_coeffs, rank_offset):
        """
        x_entries: list of 4 tuples (coeff, (i,k)) for X[0,0], X[0,1], X[1,0], X[1,1]
        y_entries: list of 4 tuples (coeff, (k,j)) for Y[0,0], Y[0,1], Y[1,0], Y[1,1]
        w_coeffs: list of 4 tuples (coeff, (i,j)) for Z[0,0], Z[0,1], Z[1,0], Z[1,1]
        """
        x00, x01, x10, x11 = x_entries
        y00, y01, y10, y11 = y_entries
        z00, z01, z10, z11 = w_coeffs
        
        # p1 = (x00+x11)*(y00+y11)
        # p2 = (x10+x11)*y00
        # p3 = x00*(y01-y11)
        # p4 = x11*(y10-y00)
        # p5 = (x00+x01)*y11
        # p6 = (x10-x00)*(y00+y01)
        # p7 = (x01-x11)*(y10+y11)
        
        # Z00 = p1+p4-p5+p7
        # Z01 = p3+p5
        # Z10 = p2+p4
        # Z11 = p1-p2+p3+p6
        
        # For each scalar product, add to U, V, W
        # p1: U has x00+x11, V has y00+y11, W has z00+z11 (coeff 1 for Z00, 1 for Z11)
        # Actually, W coefficients depend on the block-level combination
        
        # Let me handle this more carefully. For each of the 7 scalar products,
        # I need to determine the U, V, W contributions.
        
        # p1 = (x00+x11)*(y00+y11)
        # U contribution: x00 + x11
        # V contribution: y00 + y11
        # W contribution: z00 + z11 (since Z00 += p1, Z11 += p1)
        
        # p2 = (x10+x11)*y00
        # U: x10+x11, V: y00, W: z10 (since Z10 += p2)
        
        # p3 = x00*(y01-y11)
        # U: x00, V: y01-y11, W: z01 (since Z01 += p3)
        
        # p4 = x11*(y10-y00)
        # U: x11, V: y10-y00, W: z00+z10 (since Z00 += p4, Z10 += p4)
        
        # p5 = (x00+x01)*y11
        # U: x00+x01, V: y11, W: -z00+z01 (since Z00 -= p5, Z01 += p5)
        
        # p6 = (x10-x00)*(y00+y01)
        # U: x10-x00, V: y00+y01, W: z11 (since Z11 += p6)
        
        # p7 = (x01-x11)*(y10+y11)
        # U: x01-x11, V: y10+y11, W: z00 (since Z00 += p7)
        
        products = [
            # (u_entries, v_entries, w_entries)
            # p1
            ([(c, idx) for c, idx in x00] + [(c, idx) for c, idx in x11],
             [(c, idx) for c, idx in y00] + [(c, idx) for c, idx in y11],
             [(c, idx) for c, idx in z00] + [(c, idx) for c, idx in z11]),
            # p2
            ([(c, idx) for c, idx in x10] + [(c, idx) for c, idx in x11],
             y00,
             z10),
            # p3
            (x00,
             [(c, idx) for c, idx in y01] + [(-c, idx) for c, idx in y11],
             z01),
            # p4
            (x11,
             [(c, idx) for c, idx in y10] + [(-c, idx) for c, idx in y00],
             [(c, idx) for c, idx in z00] + [(c, idx) for c, idx in z10]),
            # p5
            ([(c, idx) for c, idx in x00] + [(c, idx) for c, idx in x01],
             y11,
             [(-c, idx) for c, idx in z00] + [(c, idx) for c, idx in z01]),
            # p6
            ([(c, idx) for c, idx in x10] + [(-c, idx) for c, idx in x00],
             [(c, idx) for c, idx in y00] + [(c, idx) for c, idx in y01],
             z11),
            # p7
            ([(c, idx) for c, idx in x01] + [(-c, idx) for c, idx in x11],
             [(c, idx) for c, idx in y10] + [(c, idx) for c, idx in y11],
             z00),
        ]
        
        for u_entries, v_entries, w_entries in products:
            for coeff, (i, k) in u_entries:
                U[i*4+k][rank_offset] += coeff
            for coeff, (k, j) in v_entries:
                V[k*4+j][rank_offset] += coeff
            for coeff, (i, j) in w_entries:
                W[j*4+i][rank_offset] += coeff
            rank_offset += 1
        
        return rank_offset
    
    # Define the 7 block-level products
    # A block entries as (coeff, (i,k))
    # B block entries as (coeff, (k,j))
    
    # A00: (1,(0,0)), (1,(0,1)), (1,(1,0)), (1,(1,1))
    # A01: (1,(0,2)), (1,(0,3)), (1,(1,2)), (1,(1,3))
    # A10: (1,(2,0)), (1,(2,1)), (1,(3,0)), (1,(3,1))
    # A11: (1,(2,2)), (1,(2,3)), (1,(3,2)), (1,(3,3))
    
    # B00: (1,(0,0)), (1,(0,1)), (1,(1,0)), (1,(1,1))
    # B01: (1,(0,2)), (1,(0,3)), (1,(1,2)), (1,(1,3))
    # B10: (1,(2,0)), (1,(2,1)), (1,(3,0)), (1,(3,1))
    # B11: (1,(2,2)), (1,(2,3)), (1,(3,2)), (1,(3,3))
    
    A00 = [(1,(0,0)), (1,(0,1)), (1,(1,0)), (1,(1,1))]
    A01 = [(1,(0,2)), (1,(0,3)), (1,(1,2)), (1,(1,3))]
    A10 = [(1,(2,0)), (1,(2,1)), (1,(3,0)), (1,(3,1))]
    A11 = [(1,(2,2)), (1,(2,3)), (1,(3,2)), (1,(3,3))]
    
    B00 = [(1,(0,0)), (1,(0,1)), (1,(1,0)), (1,(1,1))]
    B01 = [(1,(0,2)), (1,(0,3)), (1,(1,2)), (1,(1,3))]
    B10 = [(1,(2,0)), (1,(2,1)), (1,(3,0)), (1,(3,1))]
    B11 = [(1,(2,2)), (1,(2,3)), (1,(3,2)), (1,(3,3))]
    
    # W block coefficients for each block-level product
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # For M1: contributes to C00 (coeff 1) and C11 (coeff 1)
    # C00 block: Z[0,0] of C, which is C[0,0],C[0,1],C[1,0],C[1,1]
    # C11 block: Z[0,0] of C11, which is C[2,2],C[2,3],C[3,2],C[3,3]
    
    # M1 w_coeffs: C00 entries with coeff 1, C11 entries with coeff 1
    M1_w = [
        [(1,(0,0)), (1,(0,1)), (1,(1,0)), (1,(1,1))],  # z00: C00
        [(1,(0,2)), (1,(0,3)), (1,(1,2)), (1,(1,3))],  # z01: C01... 
    ]
    
    # Wait, I need to be more careful. Each Mi is a 2x2 matrix. 
    # M1 contributes to C00 and C11.
    # C00 is the top-left 2x2 block of C: C[0,0], C[0,1], C[1,0], C[1,1]
    # C11 is the bottom-right 2x2 block of C: C[2,2], C[2,3], C[3,2], C[3,3]
    
    # For M1, the 2x2 result is added to C00 and C11.
    # So z00 = C00 (coeff 1), z01 = C01 (coeff 0), z10 = C10 (coeff 0), z11 = C11 (coeff 1)
    # Wait no. M1 is a 2x2 matrix. Its (0,0) entry goes to C00[0,0] = C[0,0]
    # M1's (0,1) entry goes to C00[0,1] = C[0,1]
    # M1's (1,0) entry goes to C00[1,0] = C[1,0]
    # M1's (1,1) entry goes to C00[1,1] = C[1,1]
    # And also to C11[0,0] = C[2,2], C11[0,1] = C[2,3], C11[1,0] = C[3,2], C11[1,1] = C[3,3]
    
    # So for M1, the W coefficients are:
    # z00 (M1[0,0]): C[0,0] with coeff 1, C[2,2] with coeff 1
    # z01 (M1[0,1]): C[0,1] with coeff 1, C[2,3] with coeff 1
    # z10 (M1[1,0]): C[1,0] with coeff 1, C[3,2] with coeff 1
    # z11 (M1[1,1]): C[1,1] with coeff 1, C[3,3] with coeff 1
    
    # Let me define a helper to create W coefficients for a block
    def block_w(block_row, block_col, coeff=1):
        """Create W entries for a 2x2 block at position (block_row, block_col)"""
        base_i = block_row * 2
        base_j = block_col * 2
        return [
            (coeff, (base_i, base_j)),
            (coeff, (base_i, base_j+1)),
            (coeff, (base_i+1, base_j)),
            (coeff, (base_i+1, base_j+1)),
        ]
    
    # M1: C00 (coeff 1) and C11 (coeff 1)
    M1_w = [
        block_w(0,0,1) + block_w(1,1,1),  # z00
        block_w(0,1,1) + block_w(1,1,1)[1:],  # z01 - wait this is getting messy
    ]
    
    # Let me redo this. For each block-level product Mi, I need to specify
    # the W coefficients for its 4 entries (z00, z01, z10, z11).
    
    # M1 = (A00+A11)*(B00+B11), contributes +1 to C00 and +1 to C11
    # M1[0,0] -> C[0,0] + C[2,2]
    # M1[0,1] -> C[0,1] + C[2,3]
    # M1[1,0] -> C[1,0] + C[3,2]
    # M1[1,1] -> C[1,1] + C[3,3]
    
    M1_w = [
        [(1,(0,0)), (1,(2,2))],
        [(1,(0,1)), (1,(2,3))],
        [(1,(1,0)), (1,(3,2))],
        [(1,(1,1)), (1,(3,3))],
    ]
    
    # M2 = (A10+A11)*B00, contributes +1 to C10 and -1 to C11
    # M2[0,0] -> C[2,0] - C[2,2]
    # M2[0,1] -> C[2,1] - C[2,3]
    # M2[1,0] -> C[3,0] - C[3,2]
    # M2[1,1] -> C[3,1] - C[3,3]
    
    M2_w = [
        [(1,(2,0)), (-1,(2,2))],
        [(1,(2,1)), (-1,(2,3))],
        [(1,(3,0)), (-1,(3,2))],
        [(1,(3,1)), (-1,(3,3))],
    ]
    
    # M3 = A00*(B01-B11), contributes +1 to C01 and +1 to C11
    # M3[0,0] -> C[0,2] + C[2,2]
    # M3[0,1] -> C[0,3] + C[2,3]
    # M3[1,0] -> C[1,2] + C[3,2]
    # M3[1,1] -> C[1,3] + C[3,3]
    
    M3_w = [
        [(1,(0,2)), (1,(2,2))],
        [(1,(0,3)), (1,(2,3))],
        [(1,(1,2)), (1,(3,2))],
        [(1,(1,3)), (1,(3,3))],
    ]
    
    # M4 = A11*(B10-B00), contributes +1 to C00 and +1 to C10
    # M4[0,0] -> C[0,0] + C[2,0]
    # M4[0,1] -> C[0,1] + C[2,1]
    # M4[1,0] -> C[1,0] + C[3,0]
    # M4[1,1] -> C[1,1] + C[3,1]
    
    M4_w = [
        [(1,(0,0)), (1,(2,0))],
        [(1,(0,1)), (1,(2,1))],
        [(1,(1,0)), (1,(3,0))],
        [(1,(1,1)), (1,(3,1))],
    ]
    
    # M5 = (A00+A01)*B11, contributes -1 to C00 and +1 to C01
    # M5[0,0] -> -C[0,0] + C[0,2]
    # M5[0,1] -> -C[0,1] + C[0,3]
    # M5[1,0] -> -C[1,0] + C[1,2]
    # M5[1,1] -> -C[1,1] + C[1,3]
    
    M5_w = [
        [(-1,(0,0)), (1,(0,2))],
        [(-1,(0,1)), (1,(0,3))],
        [(-1,(1,0)), (1,(1,2))],
        [(-1,(1,1)), (1,(1,3))],
    ]
    
    # M6 = (A10-A00)*(B00+B01), contributes +1 to C11
    # M6[0,0] -> C[2,2]
    # M6[0,1] -> C[2,3]
    # M6[1,0] -> C[3,2]
    # M6[1,1] -> C[3,3]
    
    M6_w = [
        [(1,(2,2))],
        [(1,(2,3))],
        [(1,(3,2))],
        [(1,(3,3))],
    ]
    
    # M7 = (A01-A11)*(B10+B11), contributes +1 to C00
    # M7[0,0] -> C[0,0]
    # M7[0,1] -> C[0,1]
    # M7[1,0] -> C[1,0]
    # M7[1,1] -> C[1,1]
    
    M7_w = [
        [(1,(0,0))],
        [(1,(0,1))],
        [(1,(1,0))],
        [(1,(1,1))],
    ]
    
    # Now define the X and Y entries for each block product
    # X entries: (coeff, (i,k)) for A
    # Y entries: (coeff, (k,j)) for B
    
    # M1: X = A00+A11, Y = B00+B11
    M1_x = [
        [(1,(0,0)), (1,(2,2))],  # x00
        [(1,(0,1)), (1,(2,3))],  # x01
        [(1,(1,0)), (1,(3,2))],  # x10
        [(1,(1,1)), (1,(3,3))],  # x11
    ]
    M1_y = [
        [(1,(0,0)), (1,(2,2))],  # y00
        [(1,(0,1)), (1,(2,3))],  # y01
        [(1,(1,0)), (1,(3,2))],  # y10
        [(1,(1,1)), (1,(3,3))],  # y11
    ]
    
    # M2: X = A10+A11, Y = B00
    M2_x = [
        [(1,(2,0)), (1,(2,2))],  # x00
        [(1,(2,1)), (1,(2,3))],  # x01
        [(1,(3,0)), (1,(3,2))],  # x10
        [(1,(3,1)), (1,(3,3))],  # x11
    ]
    M2_y = [
        [(1,(0,0))],  # y00
        [(1,(0,1))],  # y01
        [(1,(1,0))],  # y10
        [(1,(1,1))],  # y11
    ]
    
    # M3: X = A00, Y = B01-B11
    M3_x = [
        [(1,(0,0))],  # x00
        [(1,(0,1))],  # x01
        [(1,(1,0))],  # x10
        [(1,(1,1))],  # x11
    ]
    M3_y = [
        [(1,(0,2)), (-1,(2,2))],  # y00
        [(1,(0,3)), (-1,(2,3))],  # y01
        [(1,(1,2)), (-1,(3,2))],  # y10
        [(1,(1,3)), (-1,(3,3))],  # y11
    ]
    
    # M4: X = A11, Y = B10-B00
    M4_x = [
        [(1,(2,2))],  # x00
        [(1,(2,3))],  # x01
        [(1,(3,2))],  # x10
        [(1,(3,3))],  # x11
    ]
    M4_y = [
        [(1,(2,0)), (-1,(0,0))],  # y00
        [(1,(2,1)), (-1,(0,1))],  # y01
        [(1,(3,0)), (-1,(1,0))],  # y10
        [(1,(3,1)), (-1,(1,1))],  # y11
    ]
    
    # M5: X = A00+A01, Y = B11
    M5_x = [
        [(1,(0,0)), (1,(0,2))],  # x00
        [(1,(0,1)), (1,(0,3))],  # x01
        [(1,(1,0)), (1,(1,2))],  # x10
        [(1,(1,1)), (1,(1,3))],  # x11
    ]
    M5_y = [
        [(1,(2,2))],  # y00
        [(1,(2,3))],  # y01
        [(1,(3,2))],  # y10
        [(1,(3,3))],  # y11
    ]
    
    # M6: X = A10-A00, Y = B00+B01
    M6_x = [
        [(1,(2,0)), (-1,(0,0))],  # x00
        [(1,(2,1)), (-1,(0,1))],  # x01
        [(1,(3,0)), (-1,(1,0))],  # x10
        [(1,(3,1)), (-1,(1,1))],  # x11
    ]
    M6_y = [
        [(1,(0,0)), (1,(0,2))],  # y00
        [(1,(0,1)), (1,(0,3))],  # y01
        [(1,(1,0)), (1,(1,2))],  # y10
        [(1,(1,1)), (1,(1,3))],  # y11
    ]
    
    # M7: X = A01-A11, Y = B10+B11
    M7_x = [
        [(1,(0,2)), (-1,(2,2))],  # x00
        [(1,(0,3)), (-1,(2,3))],  # x01
        [(1,(1,2)), (-1,(3,2))],  # x10
        [(1,(1,3)), (-1,(3,3))],  # x11
    ]
    M7_y = [
        [(1,(2,0)), (1,(2,2))],  # y00
        [(1,(2,1)), (1,(2,3))],  # y01
        [(1,(3,0)), (1,(3,2))],  # y10
        [(1,(3,1)), (1,(3,3))],  # y11
    ]
    
    block_data = [
        (M1_x, M1_y, M1_w),
        (M2_x, M2_y, M2_w),
        (M3_x, M3_y, M3_w),
        (M4_x, M4_y, M4_w),
        (M5_x, M5_y, M5_w),
        (M6_x, M6_y, M6_w),
        (M7_x, M7_y, M7_w),
    ]
    
    rank_offset = 0
    for x_entries, y_entries, w_entries in block_data:
        rank_offset = add_strassen(x_entries, y_entries, w_entries, rank_offset)
    
    return {"U": U, "V": V, "W": W}

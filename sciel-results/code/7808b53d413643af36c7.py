import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 68
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # 1. C_TL = A_top B_left (2x4x3)
    # A_top is rows 0,1 of A. B_left is cols 0,1,2 of B.
    # Split A_top into A_t1 (2x2, cols 0,1) and A_t2 (2x2, cols 2,3).
    # Split B_left into B_l1 (2x3, rows 0,1) and B_l2 (2x3, rows 2,3).
    # C_TL = A_t1 B_l1 + A_t2 B_l2.
    
    # Block 1: A_t1 B_l1
    # A_t1 indices: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,4,5
    # B_l1 indices: B[0,0..2], B[1,0..2] -> V indices 0..2, 6..8
    # C_TL part: C[0,0..2], C[1,0..2] -> W indices 0..2, 6..8 (since C is 3x6, row 0 is 0..5, row 1 is 6..11)
    # Wait, W index is k*n + i.
    # C is 3x6.
    # C[0,0] -> W[0*3+0] = 0
    # C[0,1] -> W[1*3+0] = 3
    # C[0,2] -> W[2*3+0] = 6
    # C[1,0] -> W[0*3+1] = 1
    # C[1,1] -> W[1*3+1] = 4
    # C[1,2] -> W[2*3+1] = 7
    
    # We need a function to add a 2x2x3 block with specific indices.
    def add_2x2x3_block(a_indices, b_indices, c_indices):
        # a_indices: [a00, a01, a10, a11] (U indices)
        # b_indices: [b00, b01, b02, b10, b11, b12] (V indices)
        # c_indices: [c00, c01, c02, c10, c11, c12] (W indices)
        
        a00, a01, a10, a11 = a_indices
        b00, b01, b02, b10, b11, b12 = b_indices
        c00, c01, c02, c10, c11, c12 = c_indices
        
        # Strassen 7 terms for cols 0,1
        # M1 = (a00+a01)(b00+b10)
        add_term(a00, b00, 0, 1, 1, 0) # Placeholder, need to sum into W
        # This approach of adding terms one by one is hard to map to W directly if W has multiple contributions.
        # Better to accumulate W values.
        
        # Let's use a local list for W contributions for this block.
        # But we are writing directly to global W.
        # We need to add the coefficients to the specific W indices.
        
        # M1 = (a00+a01)(b00+b10)
        # U: a00, a01. V: b00, b10.
        # This is not a single term in the tensor decomposition sense unless we expand.
        # The tensor decomposition requires linear combinations in U, V, W.
        # So U[k] can be a sum of A elements.
        
        # Let's define the 11 terms explicitly.
        # Term 1: (a00+a01)(b00+b10)
        # U part: a00+a01. V part: b00+b10.
        # We add 1 to U[a00], 1 to U[a01], 1 to V[b00], 1 to V[b10].
        # W part: contributes to c00, c01, c10, c11.
        # c00 += M1
        # c01 += 0
        # c10 += 0
        # c11 += M1
        # Wait, Strassen:
        # C11 = M1 - M2 - M3 + M7
        # C12 = M3 + M5
        # C21 = M6 + M7
        # C22 = M1 - M4 + M5 - M6
        
        # So for each M, we add its coefficient to the relevant W indices.
        
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            # u_coeffs: dict {u_idx: coeff}
            # v_coeffs: dict {v_idx: coeff}
            # w_coeffs: dict {w_idx: coeff}
            nonlocal col
            for u_idx, c in u_coeffs.items():
                U[u_idx][col] += c
            for v_idx, c in v_coeffs.items():
                V[v_idx][col] += c
            for w_idx, c in w_coeffs.items():
                W[w_idx][col] += c
            col += 1

        # M1 = (a00+a01)(b00+b10)
        add_strassen_term(
            {a00: 1, a01: 1},
            {b00: 1, b10: 1},
            {c00: 1, c11: 1} # M1 contributes +1 to C11 and C22? No.
            # C11 = M1 - M2 - M3 + M7. So M1 coeff is +1.
            # C22 = M1 - M4 + M5 - M6. So M1 coeff is +1.
            # C12, C21 do not use M1.
        )
        
        # M2 = a01(b00+b10)
        add_strassen_term(
            {a01: 1},
            {b00: 1, b10: 1},
            {c00: -1} # C11 has -M2
        )
        
        # M3 = a00(b01-b11)
        add_strassen_term(
            {a00: 1},
            {b01: 1, b11: -1},
            {c00: -1, c01: 1} # C11 has -M3, C12 has +M3
        )
        
        # M4 = a11(b02+b12) -- Wait, Strassen is for 2x2x2.
        # The 2x2x2 part uses b00, b01, b10, b11.
        # M4 = a11(b01+b11)? No.
        # Standard Strassen:
        # M1 = (a11+a12)(b11+b12)
        # M2 = a12(b11+b12)
        # M3 = a11(b21-b22)
        # M4 = a22(b11+b21)
        # M5 = (a11+a22)(b22-b11)
        # M6 = (a21+a22)b11
        # M7 = (a11-a21)(b12-b21)
        
        # My indices:
        # a00=a11, a01=a12, a10=a21, a11=a22.
        # b00=b11, b01=b12, b10=b21, b11=b22.
        
        # M1 = (a00+a01)(b00+b10) -> Done.
        # M2 = a01(b00+b10) -> Done.
        # M3 = a00(b10-b11) -> Done.
        # M4 = a11(b00+b10)
        add_strassen_term(
            {a11: 1},
            {b00: 1, b10: 1},
            {c11: -1} # C22 has -M4
        )
        
        # M5 = (a00+a11)(b11-b00)
        add_strassen_term(
            {a00: 1, a11: 1},
            {b11: 1, b00: -1},
            {c01: 1, c11: 1} # C12 has +M5, C22 has +M5
        )
        
        # M6 = (a10+a11)b00
        add_strassen_term(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1, c11: -1} # C21 has +M6, C22 has -M6
        )
        
        # M7 = (a00-a10)(b01-b10)
        add_strassen_term(
            {a00: 1, a10: -1},
            {b01: 1, b10: -1},
            {c00: 1, c10: 1} # C11 has +M7, C21 has +M7
        )
        
        # Now for the 3rd column (b02, b12).
        # We use 4 terms:
        # M8 = a00 b02
        # M9 = a01 b12
        # M10 = a10 b02
        # M11 = a11 b12
        # C02 = M8 + M9
        # C12 = M10 + M11
        
        add_strassen_term(
            {a00: 1},
            {b02: 1},
            {c02: 1}
        )
        
        add_strassen_term(
            {a01: 1},
            {b12: 1},
            {c02: 1}
        )
        
        add_strassen_term(
            {a10: 1},
            {b02: 1},
            {c12: 1}
        )
        
        add_strassen_term(
            {a11: 1},
            {b12: 1},
            {c12: 1}
        )

    # Indices for Block 1: A_t1 B_l1
    # A_t1: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B_l1: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=6, B[1,1]=7, B[1,2]=8
    # C_TL part 1: C[0,0]=0, C[0,1]=3, C[0,2]=6, C[1,0]=1, C[1,1]=4, C[1,2]=7
    add_2x2x3_block([0, 1, 4, 5], [0, 1, 2, 6, 7, 8], [0, 3, 6, 1, 4, 7])
    
    # Block 2: A_t2 B_l2
    # A_t2: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B_l2: B[2,0]=12, B[2,1]=13, B[2,2]=14, B[3,0]=18, B[3,1]=19, B[3,2]=20
    # C_TL part 2: Same C indices as Block 1 (since it's a sum)
    add_2x2x3_block([2, 3, 6, 7], [12, 13, 14, 18, 19, 20], [0, 3, 6, 1, 4, 7])
    
    # 2. C_TR = A_top B_right (2x4x3)
    # B_right is cols 3,4,5 of B.
    # Block 3: A_t1 B_r1
    # A_t1: [0, 1, 4, 5]
    # B_r1: B[0,3]=3, B[0,4]=4, B[0,5]=5, B[1,3]=9, B[1,4]=10, B[1,5]=11
    # C_TR part: C[0,3]=9, C[0,4]=10, C[0,5]=11, C[1,3]=12, C[1,4]=13, C[1,5]=14
    add_2x2x3_block([0, 1, 4, 5], [3, 4, 5, 9, 10, 11], [9, 10, 11, 12, 13, 14])
    
    # Block 4: A_t2 B_r2
    # A_t2: [2, 3, 6, 7]
    # B_r2: B[2,3]=15, B[2,4]=16, B[2,5]=17, B[3,3]=21, B[3,4]=22, B[3,5]=23
    # C_TR part: Same C indices
    add_2x2x3_block([2, 3, 6, 7], [15, 16, 17, 21, 22, 23], [9, 10, 11, 12, 13, 14])
    
    # 3. C_BL = A_bottom B_left (1x4x3)
    # A_bottom: A[2,0..3] -> U indices 8,9,10,11
    # B_left: B[0..3, 0..2] -> V indices 0..2, 6..8, 12..14, 18..20
    # C_BL: C[2,0..2] -> W indices 12,13,14
    
    for j in range(4):
        for k in range(3):
            u_idx = 8 + j
            v_idx = j * 6 + k
            w_idx = 12 + k
            add_term(u_idx, v_idx, w_idx)
            
    # 4. C_BR = A_bottom B_right (1x4x3)
    # A_bottom: U indices 8,9,10,11
    # B_right: B[0..3, 3..5] -> V indices 3..5, 9..11, 15..17, 21..23
    # C_BR: C[2,3..5] -> W indices 15,16,17
    
    for j in range(4):
        for k in range(3):
            u_idx = 8 + j
            v_idx = j * 6 + (3 + k)
            w_idx = 15 + k
            add_term(u_idx, v_idx, w_idx)
            
    return {"U": U, "V": V, "W": W}

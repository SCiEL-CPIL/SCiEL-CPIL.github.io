import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part + 16 for 4x4x1 part = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_indices: list of (w_idx, coeff)
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        # u_coeffs, v_coeffs, w_coeffs are lists of (index, coeff)
        # Since each term is rank 1, we just set the values
        # But wait, the "term" is a scalar multiplication u_val * v_val.
        # u_val is a linear combo of A entries.
        # v_val is a linear combo of B entries.
        # The product is added to C with some weights.
        
        # We need to store the coefficients for this specific scalar multiplication column
        for (idx, coeff) in u_coeffs:
            U[idx][col] += coeff
        for (idx, coeff) in v_coeffs:
            V[idx][col] += coeff
        for (idx, coeff) in w_coeffs:
            W[idx][col] += coeff
        col += 1

    # --- Part 1: 4x4x4 using Recursive Strassen ---
    
    # Define outer Strassen terms
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A blocks:
    # A11: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5 in U (i*m+j)
    # A12: A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2,3,6,7
    # A21: A[2,0], A[2,1], A[3,0], A[3,1] -> indices 8,9,12,13
    # A22: A[2,2], A[2,3], A[3,2], A[3,3] -> indices 10,11,14,15
    
    # B blocks (first 4 cols):
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,4,5 in V (j*p+k)
    # B12: B[0,2], B[0,3], B[1,2], B[1,3] -> indices 2,3,6,7
    # B21: B[2,0], B[2,1], B[3,0], B[3,1] -> indices 8,9,12,13
    # B22: B[2,2], B[2,3], B[3,2], B[3,3] -> indices 10,11,14,15
    
    # C blocks:
    # C11: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,4,1,5 in W (k*n+i)
    # C12: C[0,2], C[0,3], C[1,2], C[1,3] -> indices 8,12,9,13
    # C21: C[2,0], C[2,1], C[3,0], C[3,1] -> indices 16,20,17,21
    # C22: C[2,2], C[2,3], C[3,2], C[3,3] -> indices 24,28,25,29
    
    # Strassen Inner Formulas
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # S1 = (x00+x11)(y00+y11)
    # S2 = (x10+x11)y00
    # S3 = x00(y01-y11)
    # S4 = x11(y10-y00)
    # S5 = (x00+x01)y11
    # S6 = (x10-x00)(y00+y01)
    # S7 = (x01-x11)(y10+y11)
    
    # M entries:
    # M00 = S1 + S4 - S5 + S7
    # M01 = S3 + S5
    # M10 = S2 + S4
    # M11 = S1 - S2 + S3 + S6
    
    # C contributions for each M_i:
    # C11 += M1 + M4 - M5 + M7
    # C12 += M3 + M5
    # C21 += M2 + M4
    # C22 += M1 - M2 + M3 + M6
    
    # We will iterate through the 7 outer terms.
    # For each, we define the X and Y matrices as linear combos of A and B entries.
    # Then we decompose X*Y using the 7 inner Strassen terms.
    
    # Map A block indices to U indices
    # A11: [0, 1, 4, 5] (row 0: 0,1; row 1: 4,5)
    # A12: [2, 3, 6, 7]
    # A21: [8, 9, 12, 13]
    # A22: [10, 11, 14, 15]
    
    # Map B block indices to V indices
    # B11: [0, 1, 4, 5]
    # B12: [2, 3, 6, 7]
    # B21: [8, 9, 12, 13]
    # B22: [10, 11, 14, 15]
    
    # Map C block indices to W indices
    # C11: [0, 1, 4, 5] (k=0: 0,1; k=1: 4,5) -> W[k*n+i]
    # C12: [8, 9, 12, 13] (k=2: 8,9; k=3: 12,13)
    # C21: [16, 17, 20, 21] (k=4: 16,17; k=5: 20,21)
    # C22: [24, 25, 28, 29] (k=6: 24,25; k=7: 28,29)
    
    # Wait, W index is k*n + i.
    # C is 5x4.
    # C11 is rows 0-1, cols 0-1. k=0,1. i=0,1.
    # k=0: i=0 -> 0, i=1 -> 1.
    # k=1: i=0 -> 4, i=1 -> 5.
    # So C11 indices: 0, 1, 4, 5.
    # C12 is rows 0-1, cols 2-3. k=2,3. i=0,1.
    # k=2: i=0 -> 8, i=1 -> 9.
    # k=3: i=0 -> 12, i=1 -> 13.
    # So C12 indices: 8, 9, 12, 13.
    # C21 is rows 2-3, cols 0-1. k=4,5. i=0,1.
    # k=4: i=0 -> 16, i=1 -> 17.
    # k=5: i=0 -> 20, i=1 -> 21.
    # So C21 indices: 16, 17, 20, 21.
    # C22 is rows 2-3, cols 2-3. k=6,7. i=0,1.
    # k=6: i=0 -> 24, i=1 -> 25.
    # k=7: i=0 -> 28, i=1 -> 29.
    # So C22 indices: 24, 25, 28, 29.
    
    def get_u_coeffs(block, row, col_idx):
        # block is 0,1,2,3 for A11, A12, A21, A22
        # row is 0,1
        # col_idx is 0,1
        base = block * 4
        r = row
        c = col_idx
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        # Pattern:
        # block 0: 0,1,4,5
        # block 1: 2,3,6,7
        # block 2: 8,9,12,13
        # block 3: 10,11,14,15
        # Index = block*4 + row*2 + col?
        # 0*4 + 0*2 + 0 = 0. Correct.
        # 0*4 + 0*2 + 1 = 1. Correct.
        # 0*4 + 1*2 + 0 = 2. Wait, A11[1,0] is index 4.
        # My mapping was:
        # A11: [0, 1, 4, 5]
        # Row 0: 0, 1. Row 1: 4, 5.
        # So Index = block*4 + row*4//2? No.
        # Let's just use a lookup table.
        
        # U index = i*m + j
        # A11: i=0,1; j=0,1.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 4
        # i=1, j=1 -> 5
        
        # A12: i=0,1; j=2,3.
        # i=0, j=2 -> 2
        # i=0, j=3 -> 3
        # i=1, j=2 -> 6
        # i=1, j=3 -> 7
        
        # A21: i=2,3; j=0,1.
        # i=2, j=0 -> 8
        # i=2, j=1 -> 9
        # i=3, j=0 -> 12
        # i=3, j=1 -> 13
        
        # A22: i=2,3; j=2,3.
        # i=2, j=2 -> 10
        # i=2, j=3 -> 11
        # i=3, j=2 -> 14
        # i=3, j=3 -> 15
        
        # Formula:
        # i = (block // 2) * 2 + row
        # j = (block % 2) * 2 + col_idx
        i = (block // 2) * 2 + row
        j = (block % 2) * 2 + col_idx
        return [(i * m + j, 1)]

    def get_v_coeffs(block, row, col_idx):
        # Same structure for B
        i = (block // 2) * 2 + row
        j = (block % 2) * 2 + col_idx
        return [(i * p + j, 1)]

    def get_w_coeffs(block, row, col_idx):
        # C blocks:
        # C11: block 0. k=0,1. i=0,1.
        # C12: block 1. k=2,3. i=0,1.
        # C21: block 2. k=4,5. i=0,1.
        # C22: block 3. k=6,7. i=0,1.
        
        # k = (block // 2) * 4 + row?
        # C11: block 0. row 0 -> k=0,1? No, row in M is 0,1.
        # M is 2x2. M[row, col].
        # M[row, col] corresponds to C block element.
        # C block is 2x2.
        # C11[0,0] is C[0,0]. k=0, i=0.
        # C11[0,1] is C[0,1]. k=1, i=0.
        # C11[1,0] is C[1,0]. k=0, i=1.
        # C11[1,1] is C[1,1]. k=1, i=1.
        
        # So for C block b (0..3) and M index (r, c):
        # k = (b // 2) * 4 + r? No.
        # C11 (b=0): k=0,1.
        # C12 (b=1): k=2,3.
        # C21 (b=2): k=4,5.
        # C22 (b=3): k=6,7.
        # So k_base = (b // 2) * 4 + (b % 2) * 2?
        # b=0: 0. k=0,1.
        # b=1: 2. k=2,3.
        # b=2: 4. k=4,5.
        # b=3: 6. k=6,7.
        # Yes. k = (b // 2) * 4 + (b % 2) * 2 + r?
        # If b=0, r=0 -> k=0. r=1 -> k=1.
        # If b=1, r=0 -> k=2. r=1 -> k=3.
        # Correct.
        
        # i = (b % 2) * 2 + c?
        # C11 (b=0): i=0,1.
        # C12 (b=1): i=0,1.
        # C21 (b=2): i=0,1.
        # C22 (b=3): i=0,1.
        # Wait, C is 5x4.
        # C11 is top-left 2x2. i=0,1.
        # C12 is top-right 2x2. i=0,1.
        # C21 is bottom-left 2x2. i=2,3.
        # C22 is bottom-right 2x2. i=2,3.
        
        # So i_base = (b % 2) * 2?
        # b=0 (C11): i=0,1.
        # b=1 (C12): i=0,1.
        # b=2 (C21): i=2,3.
        # b=3 (C22): i=2,3.
        # So i = (b % 2) * 2 + c?
        # If b=0, c=0 -> i=0. c=1 -> i=1.
        # If b=1, c=0 -> i=0. c=1 -> i=1.
        # If b=2, c=0 -> i=2. c=1 -> i=3.
        # If b=3, c=0 -> i=2. c=1 -> i=3.
        # Correct.
        
        k = (b // 2) * 4 + (b % 2) * 2 + r
        i = (b % 2) * 2 + c
        return [(k * n + i, 1)]
    
    # Define the 7 outer terms
    # Each term is (X_block_coeffs, Y_block_coeffs)
    # X is 2x2, Y is 2x2.
    # We represent X as list of 4 lists of coeffs, or just generate the 7 inner terms directly.
    
    # Let's create a function to process one outer term
    def process_outer_term(X_coeffs, Y_coeffs, w_multipliers):
        # X_coeffs: 2x2 list of lists of (u_idx, coeff)
        # Y_coeffs: 2x2 list of lists of (v_idx, coeff)
        # w_multipliers: 4x4 matrix? No.
        # The outer term M contributes to C blocks with specific signs.
        # We need to pass how M[r,c] maps to C.
        # Actually, we can just pass the 4 C blocks and their signs for this M.
        # e.g. M1: C11(+1), C12(0), C21(0), C22(+1)
        
        # Inner Strassen terms S1..S7
        # S1 = (X00+X11)(Y00+Y11)
        # S2 = (X10+X11)Y00
        # S3 = X00(Y01-Y11)
        # S4 = X11(Y10-Y00)
        # S5 = (X00+X01)Y11
        # S6 = (X10-X00)(Y00+Y01)
        # S7 = (X01-X11)(Y10+Y11)
        
        def combine_u(c1, c2, sign1=1, sign2=1):
            # c1, c2 are lists of (idx, coeff)
            res = {}
            for idx, val in c1:
                res[idx] = res.get(idx, 0) + sign1 * val
            for idx, val in c2:
                res[idx] = res.get(idx, 0) + sign2 * val
            return [(k, v) for k, v in res.items() if v != 0]
        
        def combine_v(c1, c2, sign1=1, sign2=1):
            res = {}
            for idx, val in c1:
                res[idx] = res.get(idx, 0) + sign1 * val
            for idx, val in c2:
                res[idx] = res.get(idx, 0) + sign2 * val
            return [(k, v) for k, v in res.items() if v != 0]
        
        # Generate 7 terms
        terms = []
        
        # S1
        u1 = combine_u(X_coeffs[0][0], X_coeffs[1][1])
        v1 = combine_v(Y_coeffs[0][0], Y_coeffs[1][1])
        terms.append((u1, v1))
        
        # S2
        u2 = combine_u(X_coeffs[1][0], X_coeffs[1][1])
        v2 = combine_v(Y_coeffs[0][0], [])
        terms.append((u2, v2))
        
        # S3
        u3 = combine_u(X_coeffs[0][0], [])
        v3 = combine_v(Y_coeffs[0][1], Y_coeffs[1][1], 1, -1)
        terms.append((u3, v3))
        
        # S4
        u4 = combine_u(X_coeffs[1][1], [])
        v4 = combine_v(Y_coeffs[1][0], Y_coeffs[0][0], 1, -1)
        terms.append((u4, v4))
        
        # S5
        u5 = combine_u(X_coeffs[0][0], X_coeffs[0][1])
        v5 = combine_v(Y_coeffs[1][1], [])
        terms.append((u5, v5))
        
        # S6
        u6 = combine_u(X_coeffs[1][0], X_coeffs[0][0], 1, -1)
        v6 = combine_v(Y_coeffs[0][0], Y_coeffs[0][1])
        terms.append((u6, v6))
        
        # S7
        u7 = combine_u(X_coeffs[0][1], X_coeffs[1][1], 1, -1)
        v7 = combine_v(Y_coeffs[1][0], Y_coeffs[1][1])
        terms.append((u7, v7))
        
        # M entries
        # M00 = S1 + S4 - S5 + S7
        # M01 = S3 + S5
        # M10 = S2 + S4
        # M11 = S1 - S2 + S3 + S6
        
        # For each term S_j, we need to add its contribution to W.
        # The contribution to M[r,c] is a linear combo of S's.
        # So for each S_j, we determine which M[r,c] it appears in, with what coefficient.
        # Then for each M[r,c], we determine which C block it goes to, with what sign.
        # Then we add to W.
        
        # S1 is in M00(+1), M11(+1)
        # S2 is in M10(+1), M11(-1)
        # S3 is in M01(+1), M11(+1)
        # S4 is in M00(+1), M10(+1)
        # S5 is in M00(-1), M01(+1)
        # S6 is in M11(+1)
        # S7 is in M00(+1)
        
        # w_multipliers: dict {c_block: sign}
        
        for j in range(7):
            u_j, v_j = terms[j]
            w_coeffs = []
            
            # Determine M entries for S_j
            # M00: S1(+1), S4(+1), S5(-1), S7(+1)
            # M01: S3(+1), S5(+1)
            # M10: S2(+1), S4(+1)
            # M11: S1(+1), S2(-1), S3(+1), S6(+1)
            
            m_coeffs = {
                (0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0
            }
            
            if j == 0: # S1
                m_coeffs[(0,0)] += 1
                m_coeffs[(1,1)] += 1
            elif j == 1: # S2
                m_coeffs[(1,0)] += 1
                m_coeffs[(1,1)] -= 1
            elif j == 2: # S3
                m_coeffs[(0,1)] += 1
                m_coeffs[(1,1)] += 1
            elif j == 3: # S4
                m_coeffs[(0,0)] += 1
                m_coeffs[(1,0)] += 1
            elif j == 4: # S5
                m_coeffs[(0,0)] -= 1
                m_coeffs[(0,1)] += 1
            elif j == 5: # S6
                m_coeffs[(1,1)] += 1
            elif j == 6: # S7
                m_coeffs[(0,0)] += 1
            
            # Now map M[r,c] to C blocks
            for (r, c), m_val in m_coeffs.items():
                if m_val == 0: continue
                for c_block, c_sign in w_multipliers.items():
                    if c_sign == 0: continue
                    # Get W indices for this C block and M[r,c]
                    w_idx_coeffs = get_w_coeffs(c_block, r, c)
                    for w_idx, w_coeff in w_idx_coeffs:
                        final_coeff = m_val * c_sign * w_coeff
                        if final_coeff != 0:
                            w_coeffs.append((w_idx, final_coeff))
            
            add_term(u_j, v_j, w_coeffs)
    
    # Define X and Y for each outer term
    # A11: block 0, A12: block 1, A21: block 2, A22: block 3
    # B11: block 0, B12: block 1, B21: block 2, B22: block 3
    
    def get_block_u(block):
        # Returns 2x2 list of lists of (idx, coeff)
        res = [[], [], [], []]
        for r in range(2):
            for c in range(2):
                res[r][c] = get_u_coeffs(block, r, c)
        return res
    
    def get_block_v(block):
        res = [[], [], [], []]
        for r in range(2):
            for c in range(2):
                res[r][c] = get_v_coeffs(block, r, c)
        return res
    
    def combine_blocks_u(b1, b2, s1=1, s2=1):
        res = [[], [], [], []]
        for r in range(2):
            for c in range(2):
                res[r][c] = combine_u(b1[r][c], b2[r][c], s1, s2)
        return res
    
    def combine_blocks_v(b1, b2, s1=1, s2=1):
        res = [[], [], [], []]
        for r in range(2):
            for c in range(2):
                res[r][c] = combine_v(b1[r][c], b2[r][c], s1, s2)
        return res
    
    A11_u = get_block_u(0)
    A12_u = get_block_u(1)
    A21_u = get_block_u(2)
    A22_u = get_block_u(3)
    
    B11_v = get_block_v(0)
    B12_v = get_block_v(1)
    B21_v = get_block_v(2)
    B22_v = get_block_v(3)
    
    # M1 = (A11 + A22)(B11 + B22)
    X1 = combine_blocks_u(A11_u, A22_u)
    Y1 = combine_blocks_v(B11_v, B22_v)
    process_outer_term(X1, Y1, {0: 1, 3: 1}) # C11(+), C22(+)
    
    # M2 = (A21 + A22)B11
    X2 = combine_blocks_u(A21_u, A22_u)
    Y2 = B11_v
    process_outer_term(X2, Y2, {2: 1}) # C21(+)
    
    # M3 = A11(B12 - B22)
    X3 = A11_u
    Y3 = combine_blocks_v(B12_v, B22_v, 1, -1)
    process_outer_term(X3, Y3, {1: 1, 3: 1}) # C12(+), C22(+)
    
    # M4 = A22(B21 - B11)
    X4 = A22_u
    Y4 = combine_blocks_v(B21_v, B11_v, 1, -1)
    process_outer_term(X4, Y4, {0: 1, 2: 1}) # C11(+), C21(+)
    
    # M5 = (A11 + A12)B22
    X5 = combine_blocks_u(A11_u, A12_u)
    Y5 = B22_v
    process_outer_term(X5, Y5, {0: -1, 1: 1}) # C11(-), C12(+)
    
    # M6 = (A21 - A11)(B11 + B12)
    X6 = combine_blocks_u(A21_u, A11_u, 1, -1)
    Y6 = combine_blocks_v(B11_v, B12_v)
    process_outer_term(X6, Y6, {3: 1}) # C22(+)
    
    # M7 = (A12 - A22)(B21 + B22)
    X7 = combine_blocks_u(A12_u, A22_u, 1, -1)
    Y7 = combine_blocks_v(B21_v, B22_v)
    process_outer_term(X7, Y7, {0: 1}) # C11(+)
    
    # --- Part 2: 5th Column ---
    # B5 is 4x1.
    # V indices for B5:
    # B[0,4] -> 0*p + 4 = 4
    # B[1,4] -> 1*p + 4 = 9
    # B[2,4] -> 2*p + 4 = 14
    # B[3,4] -> 3*p + 4 = 19
    
    # A is 4x4.
    # C5 is 4x1.
    # W indices for C5:
    # C[0,4] -> 4*n + 0 = 16
    # C[1,4] -> 4*n + 1 = 17
    # C[2,4] -> 4*n + 2 = 18
    # C[3,4] -> 4*n + 3 = 19
    
    # C_top = A11 v1 + A12 v2
    # C_bot = A21 v1 + A22 v2
    
    # v1 = [B[0,4], B[1,4]]
    # v2 = [B[2,4], B[3,4]]
    
    # A11 v1:
    # A11[0,0]B[0,4] + A11[0,1]B[1,4] -> C[0,4]
    # A11[1,0]B[0,4] + A11[1,1]B[1,4] -> C[1,4]
    
    # Terms:
    # t1: A[0,0] * B[0,4] -> C[0,4]
    add_term([(0, 1)], [(4, 1)], [(16, 1)])
    # t2: A[0,1] * B[1,4] -> C[0,4]
    add_term([(1, 1)], [(9, 1)], [(16, 1)])
    # t3: A[1,0] * B[0,4] -> C[1,4]
    add_term([(4, 1)], [(4, 1)], [(17, 1)])
    # t4: A[1,1] * B[1,4] -> C[1,4]
    add_term([(5, 1)], [(9, 1)], [(17, 1)])
    
    # A12 v2:
    # A12[0,0]B[2,4] + A12[0,1]B[3,4] -> C[0,4]
    # A12[1,0]B[2,4] + A12[1,1]B[3,4] -> C[1,4]
    
    # t5: A[0,2] * B[2,4] -> C[0,4]
    add_term([(2, 1)], [(14, 1)], [(16, 1)])
    # t6: A[0,3] * B[3,4] -> C[0,4]
    add_term([(3, 1)], [(19, 1)], [(16, 1)])
    # t7: A[1,2] * B[2,4] -> C[1,4]
    add_term([(6, 1)], [(14, 1)], [(17, 1)])
    # t8: A[1,3] * B[3,4] -> C[1,4]
    add_term([(7, 1)], [(19, 1)], [(17, 1)])
    
    # A21 v1:
    # A21[0,0]B[0,4] + A21[0,1]B[1,4] -> C[2,4]
    # A21[1,0]B[0,4] + A21[1,1]B[1,4] -> C[3,4]
    
    # t9: A[2,0] * B[0,4] -> C[2,4]
    add_term([(8, 1)], [(4, 1)], [(18, 1)])
    # t10: A[2,1] * B[1,4] -> C[2,4]
    add_term([(9, 1)], [(9, 1)], [(18, 1)])
    # t11: A[3,0] * B[0,4] -> C[3,4]
    add_term([(12, 1)], [(4, 1)], [(19, 1)])
    # t12: A[3,1] * B[1,4] -> C[3,4]
    add_term([(13, 1)], [(9, 1)], [(19, 1)])
    
    # A22 v2:
    # A22[0,0]B[2,4] + A22[0,1]B[3,4] -> C[2,4]
    # A22[1,0]B[2,4] + A22[1,1]B[3,4] -> C[3,4]
    
    # t13: A[2,2] * B[2,4] -> C[2,4]
    add_term([(10, 1)], [(14, 1)], [(18, 1)])
    # t14: A[2,3] * B[3,4] -> C[2,4]
    add_term([(11, 1)], [(19, 1)], [(18, 1)])
    # t15: A[3,2] * B[2,4] -> C[3,4]
    add_term([(14, 1)], [(14, 1)], [(19, 1)])
    # t16: A[3,3] * B[3,4] -> C[3,4]
    add_term([(15, 1)], [(19, 1)], [(19, 1)])
    
    return {"U": U, "V": V, "W": W}

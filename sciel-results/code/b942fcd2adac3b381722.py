"""
Rank-77 exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
Uses Strassen's algorithm for the 4x4x4 block and 4 separate Strassen decompositions 
for the 4x4x2 block.
"""
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    rank = 77
    
    # U: (n*m, rank) = (16, 77)
    # V: (m*p, rank) = (24, 77)
    # W: (p*n, rank) = (24, 77)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i, c in u_coeffs:
            U[i, col] += c
        for j, c in v_coeffs:
            V[j, col] += c
        for k, c in w_coeffs:
            W[k, col] += c
        col += 1
        
    # --- Strassen 2x2x2 Terms ---
    # Indices are (row, col) within the 2x2 block
    # S1 = (x00+x11)(y00+y11)
    # S2 = (x10+x11)y00
    # S3 = x00(y01-y11)
    # S4 = x11(y10-y00)
    # S5 = (x00+x01)y11
    # S6 = (x10-x00)(y00+y01)
    # S7 = (x01-x11)(y10+y11)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
    ]
    
    # Result mapping for 2x2 Strassen
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # --- Part 1: 4x4x4 (Cols 0-3 of B and C) ---
    # A is 4x4. B1 is 4x4. C1 is 4x4.
    # We use the "big Strassen" approach which uses 7 multiplications of 2x2x2 Strassen blocks.
    # This results in 7 * 7 = 49 terms.
    
    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)
        
    def b1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*6 + (y*2+c)
        
    def c1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        # W index is k*n + i. k is col of C, i is row of C.
        # In C1 (4x4), k is 0-3, i is 0-3.
        # block 'xy' means row block x, col block y.
        # i = x*2+r, k = y*2+c
        return (y*2+c)*4 + (x*2+r)

    # Big Strassen combinations for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]), # M1
        ([('10', 1), ('11', 1)], [('00', 1)]),            # M2
        ([('00', 1)], [('01', 1), ('11', -1)]),           # M3
        ([('11', 1)], [('10', 1), ('00', -1)]),           # M4
        ([('00', 1), ('01', 1)], [('11', 1)]),            # M5
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),# M6
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)]) # M7
    ]
    
    big_result_map = {
        '00': [(0, 1), (3, 1), (4, -1), (6, 1)], # C00 = M1 + M4 - M5 + M7
        '01': [(2, 1), (4, 1)],                  # C01 = M3 + M5
        '10': [(1, 1), (3, 1)],                  # C10 = M2 + M4
        '11': [(0, 1), (1, -1), (2, 1), (5, 1)]  # C11 = M1 - M2 + M3 + M6
    }
    
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b1_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    
                    idx = c1_idx(c_block, rc[0], rc[1])
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # --- Part 2: 4x4x2 (Cols 4-5 of B and C) ---
    # A is 4x4. B_right is 4x2. C_right is 4x2.
    # We decompose this into 4 2x2x2 multiplications.
    # C0 (rows 0-1, cols 4-5) = A00*B0 + A01*B1
    # C1 (rows 2-3, cols 4-5) = A10*B0 + A11*B1
    # Where B0 is rows 0-1 of B_right, B1 is rows 2-3 of B_right.
    
    def a_idx_2x2(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)

    def b_right_idx(block, r, c):
        # block '0' is B0 (j 0-1), '1' is B1 (j 2-3)
        # k is 4-5, so c is 0-1 mapping to 4-5
        j_base = int(block) * 2
        return (j_base+r)*6 + (4+c)

    def c_right_idx(block, r, c):
        # block '0' is C0 (i 0-1), '1' is C1 (i 2-3)
        # k is 4-5, so c is 0-1 mapping to 4-5
        i_base = int(block) * 2
        return (4+c)*4 + (i_base+r)

    # 4 multiplications:
    # 1. A00 * B0 -> contributes to C0
    # 2. A01 * B1 -> contributes to C0
    # 3. A10 * B0 -> contributes to C1
    # 4. A11 * B1 -> contributes to C1
    
    mults = [
        ('00', '0', '0'),
        ('01', '1', '0'),
        ('10', '0', '1'),
        ('11', '1', '1')
    ]
    
    for a_b, b_b, c_b in mults:
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                idx = a_idx_2x2(a_b, rx, cx)
                u_coeffs.append((idx, cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                idx = b_right_idx(b_b, ry, cy)
                v_coeffs.append((idx, cy_c))
            
            w_coeffs = []
            for rc, s_contribs in result_map.items():
                small_coeff = 0
                for s_check, s_c in s_contribs:
                    if s_check == s_idx:
                        small_coeff = s_c
                        break
                if small_coeff == 0:
                    continue
                idx = c_right_idx(c_b, rc[0], rc[1])
                w_coeffs.append((idx, small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # Verification (optional, but good for debugging)
    # if col != rank:
    #     print(f"Warning: expected rank {rank}, got {col}")
    
    return {"U": U, "V": V, "W": W}

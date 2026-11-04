import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _get_strassen_2x2_terms():
    # Returns list of 7 tuples (u_indices, v_indices, w_indices)
    # Indices are 0,1,2,3 for 2x2 matrices (row-major)
    # u_indices: list of (idx, coeff)
    terms = []
    
    # S1 = (X00 + X11) * (Y00 + Y11) -> Z00 += S1, Z11 += S1
    terms.append(([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]))
    # S2 = (X10 + X11) * Y00 -> Z10 += S2, Z11 -= S2
    terms.append(([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]))
    # S3 = X00 * (Y01 - Y11) -> Z01 += S3, Z11 += S3
    terms.append(([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]))
    # S4 = X11 * (Y10 - Y00) -> Z00 += S4, Z10 += S4
    terms.append(([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]))
    # S5 = (X00 + X01) * Y11 -> Z00 -= S5, Z01 += S5
    terms.append(([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]))
    # S6 = (X10 - X00) * (Y00 + Y01) -> Z11 += S6
    terms.append(([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]))
    # S7 = (X01 - X11) * (Y10 + Y11) -> Z00 += S7
    terms.append(([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    
    return terms

def _get_strassen_4x4_terms():
    # Returns list of 49 tuples (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (global_idx, coeff) for A (4x4)
    # v_coeffs: list of (global_idx, coeff) for B (4x4)
    # w_coeffs: list of (global_idx, coeff) for C (4x4)
    
    s2_terms = _get_strassen_2x2_terms()
    
    # 4x4 Strassen steps
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Block mapping: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # A indices: row*4+col
    # B indices: row*4+col
    # C indices: col*4+row (trace convention? No, W[k*n+i,r] means W index is col*n+row for C[row,col]? 
    # Task says: W[k*n+i,r]. n=4.
    # C is 4x8. C[row, col].
    # If we split into C1 (4x4) and C2 (4x4).
    # C1[row, col] corresponds to C[row, col].
    # W index for C1[row, col] is col * 4 + row.
    
    m_steps = [
        # (L_blocks, R_blocks, C_blocks_contrib)
        # L_blocks: list of (block_idx, coeff)
        # R_blocks: list of (block_idx, coeff)
        # C_blocks_contrib: list of (block_idx, coeff)
        
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    all_terms = []
    
    for m_idx in range(7):
        L_blocks = m_steps[m_idx][0]
        R_blocks = m_steps[m_idx][1]
        C_blocks = m_steps[m_idx][2]
        
        for s2_idx in range(7):
            u_s2, v_s2, w_s2 = s2_terms[s2_idx]
            
            u_coeffs = []
            v_coeffs = []
            w_coeffs = []
            
            # Expand U
            for u_idx, u_coeff in u_s2:
                u_row_in_2x2 = u_idx // 2
                u_col_in_2x2 = u_idx % 2
                for b_idx, b_coeff in L_blocks:
                    b_row = b_idx // 2
                    b_col = b_idx % 2
                    a_row = 2 * b_row + u_row_in_2x2
                    a_col = 2 * b_col + u_col_in_2x2
                    a_global = a_row * 4 + a_col
                    u_coeffs.append((a_global, u_coeff * b_coeff))
            
            # Expand V
            for v_idx, v_coeff in v_s2:
                v_row_in_2x2 = v_idx // 2
                v_col_in_2x2 = v_idx % 2
                for b_idx, b_coeff in R_blocks:
                    b_row = b_idx // 2
                    b_col = b_idx % 2
                    b_row_global = 2 * b_row + v_row_in_2x2
                    b_col_global = 2 * b_col + v_col_in_2x2
                    b_global = b_row_global * 4 + b_col_global
                    v_coeffs.append((b_global, v_coeff * b_coeff))
            
            # Expand W
            for w_idx, w_coeff in w_s2:
                w_row_in_2x2 = w_idx // 2
                w_col_in_2x2 = w_idx % 2
                for c_idx, c_coeff in C_blocks:
                    c_row = c_idx // 2
                    c_col = c_idx % 2
                    c_row_global = 2 * c_row + w_row_in_2x2
                    c_col_global = 2 * c_col + w_col_in_2x2
                    # W index: col * n + row
                    w_global = c_col_global * 4 + c_row_global
                    w_coeffs.append((w_global, w_coeff * c_coeff))
            
            all_terms.append((u_coeffs, v_coeffs, w_coeffs))
            
    return all_terms

def construct():
    n, m, p = DIMENSIONS
    # 4x4 * 4x8
    # Split into two 4x4 * 4x4
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    base_terms = _get_strassen_4x4_terms()
    
    # Half 1: B cols 0-3, C cols 0-3
    for i, (u_c, v_c, w_c) in enumerate(base_terms):
        col = i
        for idx, coeff in u_c:
            U[idx][col] += coeff
        for idx, coeff in v_c:
            # B global index: row * 8 + col.
            # In base_terms, v_idx is row * 4 + col (for 4x4 B).
            # We need to map this to 4x8 B.
            # B is 4x8. First half is cols 0-3.
            # v_idx in base is 0..15.
            # b_row = v_idx // 4
            # b_col = v_idx % 4
            # global_v_idx = b_row * 8 + b_col
            b_row = idx // 4
            b_col = idx % 4
            global_v_idx = b_row * 8 + b_col
            V[global_v_idx][col] += coeff
        for idx, coeff in w_c:
            # W index in base is col * 4 + row.
            # C is 4x8. First half is cols 0-3.
            # w_idx in base is 0..15.
            # c_col = w_idx // 4
            # c_row = w_idx % 4
            # global_w_idx = c_col * 4 + c_row
            # This is the same index because n=4 and we are in the first half.
            W[idx][col] += coeff
            
    # Half 2: B cols 4-7, C cols 4-7
    for i, (u_c, v_c, w_c) in enumerate(base_terms):
        col = 49 + i
        for idx, coeff in u_c:
            U[idx][col] += coeff
        for idx, coeff in v_c:
            # B global index: row * 8 + col.
            # b_row = idx // 4
            # b_col = idx % 4
            # global_v_idx = b_row * 8 + (b_col + 4)
            b_row = idx // 4
            b_col = idx % 4
            global_v_idx = b_row * 8 + (b_col + 4)
            V[global_v_idx][col] += coeff
        for idx, coeff in w_c:
            # W index: col * 4 + row.
            # c_col = idx // 4
            # c_row = idx % 4
            # global_w_idx = (c_col + 4) * 4 + c_row
            c_col = idx // 4
            c_row = idx % 4
            global_w_idx = (c_col + 4) * 4 + c_row
            W[global_w_idx][col] += coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

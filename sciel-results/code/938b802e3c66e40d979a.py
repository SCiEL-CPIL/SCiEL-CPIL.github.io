import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen bilinear forms for 2x2 matrices
    # Returns list of (U_indices, V_indices, W_indices, coeffs)
    # U_indices are (i, k) in local 2x2
    # V_indices are (k, j) in local 2x2
    # W_indices are (j, i) in local 2x2 (note order for W index)
    
    # We define the 7 forms as lists of (local_u_idx, local_v_idx, local_w_idx, coeff)
    # local_u_idx: (row, col) in X
    # local_v_idx: (row, col) in Y
    # local_w_idx: (row, col) in Z
    
    # M1 = (x00 + x11)(y00 + y11)
    # z00 += M1, z11 += M1
    # M2 = (x10 + x11) y01
    # z01 += M2
    # M3 = x00 (y10 - y11)
    # z01 += M3, z11 -= M3
    # M4 = (x00 - x11)(y00 + y01)
    # z00 += M4, z11 -= M4
    # M5 = (x01 + x11)(y10 + y11)
    # z00 -= M5, z10 += M5
    # M6 = (x00 + x01)(y00 - y10)
    # z10 += M6, z11 += M6
    # M7 = (x10 + x01)(y01 - y11)
    # z00 += M7, z01 -= M7  -- Wait, check standard Strassen formulas
    
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B12
    # M3 = A11 (B21 - B22)
    # M4 = (A11 - A22) (B11 + B12)
    # M5 = (A12 + A22) (B21 + B22)
    # M6 = (A11 + A12) (B11 - B21)
    # M7 = (A21 + A12) (B12 - B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M3
    # C21 = M6 + M5
    # C22 = M1 - M3 + M6 - M4

    # Let's define the forms as (u_terms, v_terms, w_terms)
    # u_terms: list of (local_i, local_k, coeff)
    # v_terms: list of (local_k, local_j, coeff)
    # w_terms: list of (local_j, local_i, coeff)
    
    forms = []
    
    # M1
    u1 = [(0,0,1), (1,1,1)]
    v1 = [(0,0,1), (1,1,1)]
    w1 = [(0,0,1), (1,1,1)] # C11, C22
    forms.append((u1, v1, w1))
    
    # M2
    u2 = [(1,0,1), (1,1,1)]
    v2 = [(0,1,1)]
    w2 = [(0,1,1)] # C12
    forms.append((u2, v2, w2))
    
    # M3
    u3 = [(0,0,1)]
    v3 = [(1,0,1), (1,1,-1)]
    w3 = [(0,1,1), (1,1,-1)] # C12, C22
    forms.append((u3, v3, w3))
    
    # M4
    u4 = [(0,0,1), (1,1,-1)]
    v4 = [(0,0,1), (0,1,1)]
    w4 = [(0,0,1), (1,1,-1)] # C11, C22
    forms.append((u4, v4, w4))
    
    # M5
    u5 = [(0,1,1), (1,1,1)]
    v5 = [(1,0,1), (1,1,1)]
    w5 = [(0,0,-1), (1,0,1)] # C11, C21
    forms.append((u5, v5, w5))
    
    # M6
    u6 = [(0,0,1), (0,1,1)]
    v6 = [(0,0,1), (1,0,-1)]
    w6 = [(1,0,1), (1,1,1)] # C21, C22
    forms.append((u6, v6, w6))
    
    # M7
    u7 = [(1,0,1), (0,1,1)]
    v7 = [(0,1,1), (1,1,-1)]
    w7 = [(0,0,1), (0,1,-1)] # C11, C12
    forms.append((u7, v7, w7))

    # Helper to add a block multiplication
    # block_u_offset: (row_offset, col_offset) for A
    # block_v_offset: (row_offset, col_offset) for B
    # block_w_offset: (row_offset, col_offset) for C
    # w_sign: multiplier for all W entries in this block (should be 1)
    
    def add_block(u_off, v_off, w_off, base_index):
        for f_idx in range(7):
            u_terms, v_terms, w_terms = forms[f_idx]
            idx = base_index + f_idx
            
            for (li, lk, cu) in u_terms:
                gi = u_off[0] + li
                gk = u_off[1] + lk
                U[gi * m + gk][idx] += cu
            
            for (lk, lj, cv) in v_terms:
                gk = v_off[0] + lk
                gj = v_off[1] + lj
                V[gk * p + gj][idx] += cv
            
            for (lj, li, cw) in w_terms:
                gj = w_off[0] + lj
                gi = w_off[1] + li
                W[gj * n + gi][idx] += cw

    # A_L is A[:, 0:2] -> u_off (0, 0)
    # A_R is A[:, 2:4] -> u_off (0, 2)
    
    # B_LL is B[0:2, 0:2] -> v_off (0, 0)
    # B_RL is B[2:4, 0:2] -> v_off (2, 0)
    # B_LR is B[0:2, 2:4] -> v_off (0, 2)
    # B_RR is B[2:4, 2:4] -> v_off (2, 2)
    
    # C_L is C[:, 0:2] -> w_off (0, 0)
    # C_R is C[:, 2:4] -> w_off (2, 0)
    
    # 1. A_L * B_LL -> C_L
    add_block((0, 0), (0, 0), (0, 0), 0)
    
    # 2. A_R * B_RL -> C_L
    add_block((0, 2), (2, 0), (0, 0), 7)
    
    # 3. A_L * B_LR -> C_R
    add_block((0, 0), (0, 2), (2, 0), 14)
    
    # 4. A_R * B_RR -> C_R
    add_block((0, 2), (2, 2), (2, 0), 21)
    
    return {"U": U, "V": V, "W": W}

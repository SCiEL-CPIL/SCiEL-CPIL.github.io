import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
DEADLINE = 145  # Safety margin before 150s limit

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 49 (Strassen 4x4x4) + 36 (4 * Rank(2x2x3) = 4 * 9)
    total_rank = 49 + 36
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    def check_deadline():
        if time.time() - _START_TIME > DEADLINE:
            # Return what we have so far (should be complete by now)
            U_trim = [row[:current_rank] for row in U]
            V_trim = [row[:current_rank] for row in V]
            W_trim = [row[:current_rank] for row in W]
            return {"U": U_trim, "V": V_trim, "W": W_trim}
        return None
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Same as before
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    sub_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # p0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # p1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # p2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # p3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # p4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # p5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # p6
    ]
    
    for m_idx in range(7):
        early = check_deadline()
        if early is not None:
            return early
        x_def, y_def = strassen_terms[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = sub_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff_block
            
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff_block
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            cm_coeffs = {
                (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
            }
            
            if t_idx == 0:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1:
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2:
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4:
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5:
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6:
                cm_coeffs[(0,0)] += 1.0
            
            c_block_coeffs = {}
            if m_idx == 0:
                c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
            elif m_idx == 1:
                c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
            elif m_idx == 2:
                c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
            elif m_idx == 3:
                c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
            elif m_idx == 4:
                c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
            elif m_idx == 5:
                c_block_coeffs = {(1,1): 1.0}
            elif m_idx == 6:
                c_block_coeffs = {(0,0): 1.0}
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 36) ---
    # Cols 4, 5, 6 of B.
    # B_top: rows 0-1, cols 4-6.
    # B_bot: rows 2-3, cols 4-6.
    # A_00: rows 0-1, cols 0-1.
    # A_01: rows 0-1, cols 2-3.
    # A_10: rows 2-3, cols 0-1.
    # A_11: rows 2-3, cols 2-3.
    
    # Blocks of A: (row_block, col_block)
    # 0: (0,0) -> A_00
    # 1: (0,1) -> A_01
    # 2: (1,0) -> A_10
    # 3: (1,1) -> A_11
    
    # For each A block, we multiply with the corresponding B block.
    # A_00 * B_top -> contributes to C_top (rows 0-1)
    # A_01 * B_bot -> contributes to C_top (rows 0-1)
    # A_10 * B_top -> contributes to C_bot (rows 2-3)
    # A_11 * B_bot -> contributes to C_bot (rows 2-3)
    
    def add_2x2x3(a_row_block, a_col_block, b_row_block, c_row_block):
        nonlocal current_rank
        
        early = check_deadline()
        if early is not None:
            return early
        
        # a_row_block: 0 or 1 (row offset 0 or 2)
        # a_col_block: 0 or 1 (col offset 0 or 2)
        # b_row_block: 0 or 1 (row offset 0 or 2)
        # c_row_block: 0 or 1 (row offset 0 or 2)
        
        # B cols are 4, 5, 6.
        # B index: b_row * 7 + b_col
        
        # 9 terms: 7 Strassen (for cols 4,5) + 2 Optimized (for col 6)
        
        # Strassen part (cols 4,5)
        # B_00: b_row, col 4
        # B_01: b_row, col 5
        # B_10: b_row+1, col 4
        # B_11: b_row+1, col 5
        
        # A_00: a_row, a_col
        # A_01: a_row, a_col+1
        # A_10: a_row+1, a_col
        # A_11: a_row+1, a_col+1
        
        strassen_2x2x2_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # M7
        ]
        
        for t_idx in range(7):
            early = check_deadline()
            if early is not None:
                return early
            r = current_rank
            current_rank += 1
            x_sub, y_sub = strassen_2x2x2_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            # A entries
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            # B entries (cols 4, 5)
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 4 + lj # Col 4 or 5
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            # C entries (cols 4, 5)
            # C_00 (col 4) = M1 + M4 - M5 + M7
            # C_01 (col 5) = M3 + M5
            # C_10 (col 4) = M2 + M4
            # C_11 (col 5) = M1 - M2 + M3 + M6
            
            cm_coeffs = {
                (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
            }
            
            if t_idx == 0:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1:
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2:
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4:
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5:
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6:
                cm_coeffs[(0,0)] += 1.0
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                i_final = 2 * c_row_block + li
                k_final = 4 + lj # Col 4 or 5
                w_idx = k_final * n + i_final
                w_coeffs[w_idx] += cm_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
        
        # Optimized part (col 6)
        # C_02 (col 6) = A_00 B_02 + A_01 B_12
        # C_12 (col 6) = A_10 B_02 + A_11 B_12
        # B_02: b_row, col 6
        # B_12: b_row+1, col 6
        
        # Using 2 terms:
        # M8 = (A00 + A10) B02 + (A01 + A11) B12
        # M9 = (A00 - A10) B02 + (A01 - A11) B12
        # C02 = 0.5 * (M8 + M9)
        # C12 = 0.5 * (M8 - M9)
        
        # Term 8
        r = current_rank
        current_rank += 1
        
        u_coeffs = [0.0] * (n*m)
        v_coeffs = [0.0] * (m*p)
        
        # A entries for M8: A00, A10, A01, A11 all +1
        # A00: (0,0) relative to block
        # A10: (1,0) relative
        # A01: (0,1) relative
        # A11: (1,1) relative
        
        u_coeffs[ (2*a_row_block + 0) * m + (2*a_col_block + 0) ] = 1.0
        u_coeffs[ (2*a_row_block + 1) * m + (2*a_col_block + 0) ] = 1.0
        u_coeffs[ (2*a_row_block + 0) * m + (2*a_col_block + 1) ] = 1.0
        u_coeffs[ (2*a_row_block + 1) * m + (2*a_col_block + 1) ] = 1.0
        
        # B entries for M8: B02, B12 all +1
        v_coeffs[ (2*b_row_block + 0) * p + 6 ] = 1.0
        v_coeffs[ (2*b_row_block + 1) * p + 6 ] = 1.0
        
        for i in range(n*m):
            U[i][r] = u_coeffs[i]
        for j in range(m*p):
            V[j][r] = v_coeffs[j]
        
        w_coeffs = [0.0] * (p*n)
        # C02 gets 0.5, C12 gets 0.5
        w_coeffs[ 6 * n + (2*c_row_block + 0) ] = 0.5
        w_coeffs[ 6 * n + (2*c_row_block + 1) ] = 0.5
        
        for i in range(p*n):
            W[i][r] = w_coeffs[i]
            
        early = check_deadline()
        if early is not None:
            return early

        # Term 9
        r = current_rank
        current_rank += 1
        
        u_coeffs = [0.0] * (n*m)
        v_coeffs = [0.0] * (m*p)
        
        # A entries for M9: A00, A01 +1; A10, A11 -1
        u_coeffs[ (2*a_row_block + 0) * m + (2*a_col_block + 0) ] = 1.0
        u_coeffs[ (2*a_row_block + 1) * m + (2*a_col_block + 0) ] = -1.0
        u_coeffs[ (2*a_row_block + 0) * m + (2*a_col_block + 1) ] = 1.0
        u_coeffs[ (2*a_row_block + 1) * m + (2*a_col_block + 1) ] = -1.0
        
        # B entries for M9: B02, B12 all +1
        v_coeffs[ (2*b_row_block + 0) * p + 6 ] = 1.0
        v_coeffs[ (2*b_row_block + 1) * p + 6 ] = 1.0
        
        for i in range(n*m):
            U[i][r] = u_coeffs[i]
        for j in range(m*p):
            V[j][r] = v_coeffs[j]
        
        w_coeffs = [0.0] * (p*n)
        # C02 gets 0.5, C12 gets -0.5
        w_coeffs[ 6 * n + (2*c_row_block + 0) ] = 0.5
        w_coeffs[ 6 * n + (2*c_row_block + 1) ] = -0.5
        
        for i in range(p*n):
            W[i][r] = w_coeffs[i]

    # Call for the 4 blocks
    early = add_2x2x3(0, 0, 0, 0)
    if early is not None:
        return early
    early = add_2x2x3(0, 1, 1, 0)
    if early is not None:
        return early
    early = add_2x2x3(1, 0, 0, 1)
    if early is not None:
        return early
    early = add_2x2x3(1, 1, 1, 1)
    if early is not None:
        return early

    # Trim
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

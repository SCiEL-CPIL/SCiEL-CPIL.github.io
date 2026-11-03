import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 49 (4x4x4 Strassen) + 4 * 11 (4 x 2x2x3) = 93
    # We will build up to a max rank and then trim.
    max_rank = 112
    U = [[0.0] * max_rank for _ in range(n * m)]
    V = [[0.0] * max_rank for _ in range(m * p)]
    W = [[0.0] * max_rank for _ in range(p * n)]
    
    current_rank = 0
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal current_rank
        if current_rank >= max_rank:
            return
        r = current_rank
        current_rank += 1
        for i in range(n * m):
            if u_coeffs[i] != 0:
                U[i][r] += u_coeffs[i]
        for j in range(m * p):
            if v_coeffs[j] != 0:
                V[j][r] += v_coeffs[j]
        for k in range(p * n):
            if w_coeffs[k] != 0:
                W[k][r] += w_coeffs[k]

    # --- Part 1: 4x4x4 Strassen ---
    # This part is identical to the logic in the current program, verified to be rank 49.
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
    
    # W contribution mapping for 2x2 Strassen result to C blocks
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    
    # For each M_m, and each p_t, we need to know where it goes in C.
    # We pre-compute the W coefficients for each (m_idx, t_idx).
    
    for m_idx in range(7):
        x_def, y_def = strassen_terms[m_idx]
        
        for t_idx in range(7):
            if time.time() > deadline:
                break
            x_sub, y_sub = sub_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            # U coefficients
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff_block
            
            # V coefficients (B is 4x7, we use cols 0-3)
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff_block
            
            # W coefficients
            w_coeffs = [0.0] * (p*n)
            
            # cm_coeffs: contribution of p_t to the 2x2 result matrix M_m
            cm_coeffs = {
                (0,0): 0.0,
                (0,1): 0.0,
                (1,0): 0.0,
                (1,1): 0.0
            }
            
            if t_idx == 0: # p0
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1: # p1
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2: # p2
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3: # p3
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4: # p4
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5: # p5
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6: # p6
                cm_coeffs[(0,0)] += 1.0
            
            # c_block_coeffs: contribution of M_m to the 4x4 C blocks
            c_block_coeffs = {}
            if m_idx == 0: # M1
                c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
            elif m_idx == 1: # M2
                c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
            elif m_idx == 2: # M3
                c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
            elif m_idx == 3: # M4
                c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
            elif m_idx == 4: # M5
                c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
            elif m_idx == 5: # M6
                c_block_coeffs = {(1,1): 1.0}
            elif m_idx == 6: # M7
                c_block_coeffs = {(0,0): 1.0}
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    # --- Part 2: 4x4x3 using 4x(2x2x3) ---
    # We split A into 4 2x2 blocks and B (cols 4-6) into top/bottom 2x3 blocks.
    # Sub-problems:
    # 1. A00 * B_top
    # 2. A01 * B_bot
    # 3. A10 * B_top
    # 4. A11 * B_bot
    
    # Helper to generate terms for a 2x2x3 multiplication
    # a_row_start, a_col_start: top-left of A block
    # b_row_start: top row of B block
    # b_col_start: left col of B block (always 4)
    # c_row_start, c_col_start: top-left of C block (C is 4x7, so col_start is 4)
    
    def add_2x2x3(a_rs, a_cs, b_rs, b_cs, c_rs, c_cs):
        # A block is 2x2, B block is 2x3
        # Strassen for first 2 cols of B (cols b_cs, b_cs+1)
        # Schoolbook for last 1 col of B (col b_cs+2)
        
        # Strassen terms for 2x2x2
        # Local indices for A: 0,1
        # Local indices for B: 0,1 (rows), 0,1 (cols)
        
        strassen_2x2_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # M7
        ]
        
        # W mapping for 2x2 Strassen result
        # C_00 = M1 + M4 - M5 + M7
        # C_01 = M3 + M5
        # C_10 = M2 + M4
        # C_11 = M1 - M2 + M3 + M6
        
        for m_idx in range(7):
            if time.time() > deadline:
                return
            x_sub, y_sub = strassen_2x2_terms[m_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            # U
            for (li, lj), c_x in x_sub.items():
                i = a_rs + li
                j = a_cs + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            # V (B cols b_cs, b_cs+1)
            for (li, lj), c_y in y_sub.items():
                i = b_rs + li
                j = b_cs + lj
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            # W
            w_coeffs = [0.0] * (p*n)
            
            cm_coeffs = {
                (0,0): 0.0,
                (0,1): 0.0,
                (1,0): 0.0,
                (1,1): 0.0
            }
            
            if m_idx == 0: # M1
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif m_idx == 1: # M2
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif m_idx == 2: # M3
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif m_idx == 3: # M4
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif m_idx == 4: # M5
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif m_idx == 5: # M6
                cm_coeffs[(1,1)] += 1.0
            elif m_idx == 6: # M7
                cm_coeffs[(0,0)] += 1.0
            
            # C block is 2x2, located at (c_rs, c_cs) in C
            # c_block_coeffs maps M_m to C_2x2 entries
            c_block_coeffs = {}
            if m_idx == 0: # M1
                c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
            elif m_idx == 1: # M2
                c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
            elif m_idx == 2: # M3
                c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
            elif m_idx == 3: # M4
                c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
            elif m_idx == 4: # M5
                c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
            elif m_idx == 5: # M6
                c_block_coeffs = {(1,1): 1.0}
            elif m_idx == 6: # M7
                c_block_coeffs = {(0,0): 1.0}
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    i_final = c_rs + br
                    k_final = c_cs + bc
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
        
        # Schoolbook for the 3rd column (col b_cs+2)
        # 4 terms
        for li in range(2):
            for lj in range(2):
                if time.time() > deadline:
                    return
                u_coeffs = [0.0] * (n*m)
                v_coeffs = [0.0] * (m*p)
                w_coeffs = [0.0] * (p*n)
                
                i = a_rs + li
                j = a_cs + lj
                u_idx = i * m + j
                u_coeffs[u_idx] = 1.0
                
                i_b = b_rs + li
                j_b = b_cs + 2
                v_idx = i_b * p + j_b
                v_coeffs[v_idx] = 1.0
                
                i_c = c_rs + li
                k_c = c_cs + 2
                w_idx = k_c * n + i_c
                w_coeffs[w_idx] = 1.0
                
                add_term(u_coeffs, v_coeffs, w_coeffs)

    # Call for the 4 sub-problems
    # A is 4x4. Blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    # B (cols 4-6) is 4x3.
    # B_top: rows 0-1, cols 4-6
    # B_bot: rows 2-3, cols 4-6
    # C (cols 4-6) is 4x3.
    # C_top: rows 0-1, cols 4-6
    # C_bot: rows 2-3, cols 4-6
    
    # C_top = A00 B_top + A01 B_bot
    # C_bot = A10 B_top + A11 B_bot
    
    add_2x2x3(0, 0, 0, 4, 0, 4) # A00 * B_top -> C_top
    add_2x2x3(0, 2, 2, 4, 0, 4) # A01 * B_bot -> C_top
    add_2x2x3(2, 0, 0, 4, 2, 4) # A10 * B_top -> C_bot
    add_2x2x3(2, 2, 2, 4, 2, 4) # A11 * B_bot -> C_bot

    # Trim
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")

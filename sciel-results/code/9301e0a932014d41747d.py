"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit (5s safety margin)
    
    # We will use a decomposition of rank 65.
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Helper to add a rank-1 term
    # u_indices: list of (i, j, coeff)
    # v_indices: list of (j, k, coeff)
    # w_indices: list of (k, i, coeff)
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # Split A (4x4) into 2x2 blocks: A00, A01, A10, A11
    # Split B (4x4) into 2x2 blocks: B00, B01, B10, B11
    
    # Strassen 7 multiplications for 2x2 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix product, decomposed using Strassen (7 scalar mults)
    
    # Function to get U, V, W lists for a super term
    def get_super_term_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        # Determine P and Q block coefficients
        if super_idx == 0:  # M1 = (A00+A11)(B00+B11)
            p_blocks = [1, 0, 0, 1]
            q_blocks = [1, 0, 0, 1]
        elif super_idx == 1:  # M2 = (A10+A11)B00
            p_blocks = [0, 0, 1, 1]
            q_blocks = [1, 0, 0, 0]
        elif super_idx == 2:  # M3 = A00(B01-B11)
            p_blocks = [1, 0, 0, 0]
            q_blocks = [0, 1, 0, -1]
        elif super_idx == 3:  # M4 = A11(B10-B00)
            p_blocks = [0, 0, 0, 1]
            q_blocks = [-1, 0, 1, 0]
        elif super_idx == 4:  # M5 = (A00+A01)B11
            p_blocks = [1, 1, 0, 0]
            q_blocks = [0, 0, 0, 1]
        elif super_idx == 5:  # M6 = (A10-A00)(B00+B01)
            p_blocks = [-1, 0, 1, 0]
            q_blocks = [1, 1, 0, 0]
        elif super_idx == 6:  # M7 = (A01-A11)(B10+B11)
            p_blocks = [0, 1, 0, -1]
            q_blocks = [0, 0, 1, 1]
        else:
            return u_list, v_list, w_list
            
        # Inner Strassen terms determine which P and Q elements are multiplied
        # Inner 0: (P00+P11)(Q00+Q11)
        # Inner 1: (P10+P11)Q00
        # Inner 2: P00(Q01-Q11)
        # Inner 3: P11(Q10-Q00)
        # Inner 4: (P00+P01)Q11
        # Inner 5: (P10-P00)(Q00+Q01)
        # Inner 6: (P01-P11)(Q10+Q11)
        
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
        elif inner_idx == 3:
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
        else:
            return u_list, v_list, w_list
            
        # Helper to get P element linear combo
        def get_p_element(p_elem_idx):
            res = []
            r = p_elem_idx // 2
            c = p_elem_idx % 2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0:  # A00
                    gi, gj = r, c
                elif b_idx == 1:  # A01
                    gi, gj = r, 2 + c
                elif b_idx == 2:  # A10
                    gi, gj = 2 + r, c
                elif b_idx == 3:  # A11
                    gi, gj = 2 + r, 2 + c
                res.append((gi, gj, coeff))
            return res
            
        def get_q_element(q_elem_idx):
            res = []
            r = q_elem_idx // 2
            c = q_elem_idx % 2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0:  # B00
                    gj, gk = r, c
                elif b_idx == 1:  # B01
                    gj, gk = r, 2 + c
                elif b_idx == 2:  # B10
                    gj, gk = 2 + r, c
                elif b_idx == 3:  # B11
                    gj, gk = 2 + r, 2 + c
                res.append((gj, gk, coeff))
            return res
            
        # Build u_list
        for p_idx, p_coeff in p_elems:
            for i, j, c in get_p_element(p_idx):
                u_list.append((i, j, c * p_coeff))
                
        # Build v_list
        for q_idx, q_coeff in q_elems:
            for j, k, c in get_q_element(q_idx):
                v_list.append((j, k, c * q_coeff))
                
        # Build w_list using Strassen contribution formulas
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Inner term contributions to S (2x2 result)
        if inner_idx == 0:
            s_contrib = [1, 0, 0, 1]  # S00, S01, S10, S11
        elif inner_idx == 1:
            s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2:
            s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3:
            s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4:
            s_contrib = [-1, 1, 0, 0]
        elif inner_idx == 5:
            s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6:
            s_contrib = [1, 0, 0, 0]
        
        # Super term contributions to C blocks
        if super_idx == 0:  # M1
            c_block_contrib = [1, 0, 0, 1]  # C00, C01, C10, C11
        elif super_idx == 1:  # M2
            c_block_contrib = [0, 0, 1, -1]
        elif super_idx == 2:  # M3
            c_block_contrib = [0, 1, 0, 1]
        elif super_idx == 3:  # M4
            c_block_contrib = [1, 0, 1, 0]
        elif super_idx == 4:  # M5
            c_block_contrib = [-1, 1, 0, 0]
        elif super_idx == 5:  # M6
            c_block_contrib = [0, 0, 0, 1]
        elif super_idx == 6:  # M7
            c_block_contrib = [1, 0, 0, 0]
            
        for s_pos in range(4):
            s_coeff = s_contrib[s_pos]
            if s_coeff == 0: continue
            s_r = s_pos // 2
            s_c = s_pos % 2
            for c_b in range(4):
                c_coeff = c_block_contrib[c_b]
                if c_coeff == 0: continue
                total_coeff = s_coeff * c_coeff
                if total_coeff == 0: continue
                if c_b == 0:
                    gi, gk = s_r, s_c
                elif c_b == 1:
                    gi, gk = s_r, 2 + s_c
                elif c_b == 2:
                    gi, gk = 2 + s_r, s_c
                elif c_b == 3:
                    gi, gk = 2 + s_r, 2 + s_c
                w_list.append((gk, gi, total_coeff))
                
        return u_list, v_list, w_list

    # Add all 49 terms
    for s in range(7):
        for t in range(7):
            u_list, v_list, w_list = get_super_term_indices(s, t)
            add_term(col, u_list, v_list, w_list)
            col += 1

    # --- Part 2: 4x4 * 4x1 using Schoolbook ---
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

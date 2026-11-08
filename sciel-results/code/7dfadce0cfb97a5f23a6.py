import time
import sys

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-65 tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Rank 65 is the sum of the rank of 4x4*4x4 (49, Strassen) and 4x4*4x1 (16, Schoolbook).
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
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
    
    # --- Part 1: 4x4 * 4x4 using Strassen (49 terms) ---
    
    # Strassen 7 terms for 2x2 matrices P, Q:
    # M1 = (P00+P11)(Q00+Q11)
    # M2 = (P10+P11)Q00
    # M3 = P00(Q01-Q11)
    # M4 = P11(Q10-Q00)
    # M5 = (P00+P01)Q11
    # M6 = (P10-P00)(Q00+Q01)
    # M7 = (P01-P11)(Q10+Q11)
    
    # Result C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Super-level: A (4x4) split into A00, A01, A10, A11 (2x2 blocks)
    # B (4x4) split into B00, B01, B10, B11 (2x2 blocks)
    
    # Super Strassen terms:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Result C (4x4):
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Helper to get indices for a specific super term and inner term
    def get_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        # Coefficients for A blocks in P (super term)
        # p_blocks: [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
        if super_idx == 0: p_blocks = [1, 0, 0, 1]
        elif super_idx == 1: p_blocks = [0, 0, 1, 1]
        elif super_idx == 2: p_blocks = [1, 0, 0, 0]
        elif super_idx == 3: p_blocks = [0, 0, 0, 1]
        elif super_idx == 4: p_blocks = [1, 1, 0, 0]
        elif super_idx == 5: p_blocks = [-1, 0, 1, 0]
        elif super_idx == 6: p_blocks = [0, 1, 0, -1]
        
        # Coefficients for B blocks in Q (super term)
        # q_blocks: [coeff_B00, coeff_B01, coeff_B10, coeff_B11]
        if super_idx == 0: q_blocks = [1, 0, 0, 1]
        elif super_idx == 1: q_blocks = [1, 0, 0, 0]
        elif super_idx == 2: q_blocks = [0, 1, 0, -1]
        elif super_idx == 3: q_blocks = [-1, 0, 1, 0]
        elif super_idx == 4: q_blocks = [0, 0, 0, 1]
        elif super_idx == 5: q_blocks = [1, 1, 0, 0]
        elif super_idx == 6: q_blocks = [0, 0, 1, 1]
        
        # Inner Strassen terms for 2x2 P, Q
        # P = [[P00, P01], [P10, P11]]
        # Q = [[Q00, Q01], [Q10, Q11]]
        
        # Inner term definitions (p_elems, q_elems, s_contrib)
        # p_elems: list of (elem_idx, coeff)
        # elem_idx: 0:00, 1:01, 2:10, 3:11
        
        if inner_idx == 0: # (P00+P11)(Q00+Q11)
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
            s_contrib = [1, 0, 0, 1] # S00, S01, S10, S11
        elif inner_idx == 1: # (P10+P11)Q00
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
            s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2: # P00(Q01-Q11)
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
            s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3: # P11(Q10-Q00)
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
            s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4: # (P00+P01)Q11
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
            s_contrib = [-1, 1, 0, 0]
        elif inner_idx == 5: # (P10-P00)(Q00+Q01)
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
            s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6: # (P01-P11)(Q10+Q11)
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
            s_contrib = [1, 0, 0, 0]
        
        # Build u_list
        for p_idx, p_coeff in p_elems:
            r = p_idx // 2
            c = p_idx % 2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0: gi, gj = r, c
                elif b_idx == 1: gi, gj = r, 2+c
                elif b_idx == 2: gi, gj = 2+r, c
                elif b_idx == 3: gi, gj = 2+r, 2+c
                u_list.append((gi, gj, coeff * p_coeff))
        
        # Build v_list
        for q_idx, q_coeff in q_elems:
            r = q_idx // 2
            c = q_idx % 2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0: gj, gk = r, c
                elif b_idx == 1: gj, gk = r, 2+c
                elif b_idx == 2: gj, gk = 2+r, c
                elif b_idx == 3: gj, gk = 2+r, 2+c
                v_list.append((gj, gk, coeff * q_coeff))
        
        # Build w_list
        # Map S contributions to C blocks
        # Super term contributions to C blocks:
        # C00, C01, C10, C11
        if super_idx == 0: c_block_contrib = [1, 0, 0, 1]
        elif super_idx == 1: c_block_contrib = [0, 0, 1, -1]
        elif super_idx == 2: c_block_contrib = [0, 1, 0, 1]
        elif super_idx == 3: c_block_contrib = [1, 0, 1, 0]
        elif super_idx == 4: c_block_contrib = [-1, 1, 0, 0]
        elif super_idx == 5: c_block_contrib = [0, 0, 0, 1]
        elif super_idx == 6: c_block_contrib = [1, 0, 0, 0]
        
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
                
                if c_b == 0: gi, gk = s_r, s_c
                elif c_b == 1: gi, gk = s_r, 2+s_c
                elif c_b == 2: gi, gk = 2+s_r, s_c
                elif c_b == 3: gi, gk = 2+s_r, 2+s_c
                
                w_list.append((gk, gi, total_coeff))
                
        return u_list, v_list, w_list

    for s in range(7):
        for t in range(7):
            u_list, v_list, w_list = get_indices(s, t)
            add_term(col, u_list, v_list, w_list)
            col += 1

    # --- Part 2: 4x4 * 4x1 using Schoolbook (16 terms) ---
    # A (4x4) * B_col (4x1)
    # C[i,4] = sum_j A[i,j] B[j,4]
    
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

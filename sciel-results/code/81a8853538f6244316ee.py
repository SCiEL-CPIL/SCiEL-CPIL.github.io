"""Combined approach: structured search for low-rank exact tensor decomposition."""
import numpy as np
import time
import random
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # We aim for rank 77 using:
    # 1. Strassen (rank 49) for the 4x4 * 4x4 part (B cols 0-3, C cols 0-3)
    # 2. Four 2x2x2 Strassen multiplications (rank 4*7=28) for the 4x4 * 4x2 part
    # Total rank = 49 + 28 = 77
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i, c in u_coeffs:
            if c != 0:
                U[i][col] += c
        for j, c in v_coeffs:
            if c != 0:
                V[j][col] += c
        for k, c in w_coeffs:
            if c != 0:
                W[k][col] += c
        col += 1
        # Deadline guard
        if time.time() > DEADLINE:
            return False
        return True
    
    # --- Part 1: Strassen for 4x4x4 (B cols 0-3, C cols 0-3) ---
    
    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)
        
    def b1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*6 + (y*2+c)
        
    def c1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (y*2+c)*4 + (x*2+r)

    # Strassen 2x2 terms
    strassen_2x2_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
    ]
    
    # Result mapping for 2x2 Strassen
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Big Strassen 4x4 combinations
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]),
        ([('10', 1), ('11', 1)], [('00', 1)]),
        ([('00', 1)], [('01', 1), ('11', -1)]),
        ([('11', 1)], [('10', 1), ('00', -1)]),
        ([('00', 1), ('01', 1)], [('11', 1)]),
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)])
    ]
    
    # Big Result mapping
    big_result_map = {
        '00': [(0, 1), (3, 1), (4, -1), (6, 1)],
        '01': [(2, 1), (4, 1)],
        '10': [(1, 1), (3, 1)],
        '11': [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Iterate through the 7 big multiplications
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
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
            
            if not add_term(u_coeffs, v_coeffs, w_coeffs):
                return {"U": U, "V": V, "W": W}
            
    # --- Part 2: 4x4x2 part (B cols 4-5, C cols 4-5) ---
    # Four 2x2x2 Strassen multiplications, each rank 7, total rank 28
    
    def a_idx_2x2(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)

    def b_right_idx(block, r, c):
        j_base = int(block) * 2
        return (j_base+r)*6 + (4+c)

    def c_right_idx(block, r, c):
        i_base = int(block) * 2
        return (4+c)*4 + (i_base+r)
    
    # Four multiplications:
    # 1. A00 * B0 -> C0
    # 2. A01 * B1 -> C0
    # 3. A10 * B0 -> C1
    # 4. A11 * B1 -> C1
    
    mults = [
        ('00', '0', '0'),
        ('01', '1', '0'),
        ('10', '0', '1'),
        ('11', '1', '1')
    ]
    
    for a_b, b_b, c_b in mults:
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
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
            
            if not add_term(u_coeffs, v_coeffs, w_coeffs):
                return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

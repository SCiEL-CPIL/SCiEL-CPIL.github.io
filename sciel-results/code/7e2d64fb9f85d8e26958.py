# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 65: 49 for 4x4*4x4 (Strassen) + 16 for 4x4*4x1 (Schoolbook)
    rank = 65
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j, col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k, col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i, col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # Define Strassen terms for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Super Strassen 7 terms
    super_p_blocks = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1], # M7: A01-A11
    ]
    
    super_q_blocks = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [-1, 0, 1, 0], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1],  # M7: B10+B11
    ]
    
    inner_p_elems = [
        [(0, 1), (3, 1)],  # s1: P00+P11
        [(2, 1), (3, 1)],  # s2: P10+P11
        [(0, 1)],          # s3: P00
        [(3, 1)],          # s4: P11
        [(0, 1), (1, 1)],  # s5: P00+P01
        [(2, 1), (0, -1)], # s6: P10-P00
        [(1, 1), (3, -1)], # s7: P01-P11
    ]
    
    inner_q_elems = [
        [(0, 1), (3, 1)],  # s1: Q00+Q11
        [(0, 1)],          # s2: Q00
        [(1, 1), (3, -1)], # s3: Q01-Q11
        [(2, 1), (0, -1)], # s4: Q10-Q00
        [(3, 1)],          # s5: Q11
        [(0, 1), (1, 1)],  # s6: Q00+Q01
        [(2, 1), (3, 1)],  # s7: Q10+Q11
    ]
    
    inner_s_contrib = [
        [1, 0, 0, 1],  # s1
        [0, 0, 1, -1], # s2
        [0, 1, 0, 1],  # s3
        [1, 0, 1, 0],  # s4
        [-1, 1, 0, 0], # s5
        [0, 0, 0, 1],  # s6
        [1, 0, 0, 0],  # s7
    ]
    
    super_c_block_contrib = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0],  # M7
    ]
    
    def get_p_element(p_elem_idx, p_blocks):
        res = []
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi = r; gj = c
            elif b_idx == 1:
                gi = r; gj = 2 + c
            elif b_idx == 2:
                gi = 2 + r; gj = c
            elif b_idx == 3:
                gi = 2 + r; gj = 2 + c
            res.append((gi, gj, coeff))
        return res
    
    def get_q_element(q_elem_idx, q_blocks):
        res = []
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj = r; gk = c
            elif b_idx == 1:
                gj = r; gk = 2 + c
            elif b_idx == 2:
                gj = 2 + r; gk = c
            elif b_idx == 3:
                gj = 2 + r; gk = 2 + c
            res.append((gj, gk, coeff))
        return res
    
    for s in range(7):
        for t in range(7):
            p_blocks = super_p_blocks[s]
            q_blocks = super_q_blocks[s]
            
            u_list = []
            v_list = []
            w_list = []
            
            for p_idx, p_coeff in inner_p_elems[t]:
                elems = get_p_element(p_idx, p_blocks)
                for i, j, c in elems:
                    u_list.append((i, j, c * p_coeff))
            
            for q_idx, q_coeff in inner_q_elems[t]:
                elems = get_q_element(q_idx, q_blocks)
                for j, k, c in elems:
                    v_list.append((j, k, c * q_coeff))
            
            s_contrib = inner_s_contrib[t]
            c_block_contrib = super_c_block_contrib[s]
            
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_b in range(4):
                    c_coeff = c_block_contrib[c_b]
                    if c_coeff == 0:
                        continue
                    
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    
                    if c_b == 0:
                        gi = s_r; gk = s_c
                    elif c_b == 1:
                        gi = s_r; gk = 2 + s_c
                    elif c_b == 2:
                        gi = 2 + s_r; gk = s_c
                    elif c_b == 3:
                        gi = 2 + s_r; gk = 2 + s_c
                    
                    w_list.append((gk, gi, total_coeff))
            
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

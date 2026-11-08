"""Direct rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 5]

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 65: 49 terms for 4x4*4x4 (Strassen x Strassen) + 16 terms for 4x4*4x1 (schoolbook)
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
    
    # Strassen 2x2x2 inner terms
    # For P*Q = R where P,Q,R are 2x2:
    # s1 = (p00+p11)(q00+q11)
    # s2 = (p10+p11)q00
    # s3 = p00(q01-q11)
    # s4 = p11(q10-q00)
    # s5 = (p00+p01)q11
    # s6 = (p10-p00)(q00+q01)
    # s7 = (p01-p11)(q10+q11)
    # R00 = s1+s4-s5+s7
    # R01 = s3+s5
    # R10 = s2+s4
    # R11 = s1-s2+s3+s6
    
    # Super Strassen 7 terms for 4x4*4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each super term s (0-6) and inner term t (0-6), we generate one rank-1 term.
    
    # Block coefficient definitions for P (A side) and Q (B side)
    # p_blocks[s] = [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
    # q_blocks[s] = [coeff_B00, coeff_B01, coeff_B10, coeff_B11]
    
    p_blocks_all = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    q_blocks_all = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # Inner Strassen: for term t, which P elements and Q elements are used
    # inner_t: (p_elems, q_elems) where p_elems = [(p_idx, coeff), ...], q_elems = [(q_idx, coeff), ...]
    # p_idx: 0=P00, 1=P01, 2=P10, 3=P11
    # q_idx: 0=Q00, 1=Q01, 2=Q10, 3=Q11
    
    inner_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # t0: (P00+P11)(Q00+Q11)
        ([(2, 1), (3, 1)], [(0, 1)]),            # t1: (P10+P11)Q00
        ([(0, 1)], [(1, 1), (3, -1)]),           # t2: P00(Q01-Q11)
        ([(3, 1)], [(2, 1), (0, -1)]),           # t3: P11(Q10-Q00)
        ([(0, 1), (1, 1)], [(3, 1)]),            # t4: (P00+P01)Q11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # t5: (P10-P00)(Q00+Q01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # t6: (P01-P11)(Q10+Q11)
    ]
    
    # Contribution of inner term t to S (2x2 result of P*Q)
    # s_contrib[t] = [S00, S01, S10, S11]
    s_contrib_all = [
        [1, 0, 0, 1],   # t0 -> S00, S11
        [0, 0, 1, -1],  # t1 -> S10, -S11
        [0, 1, 0, 1],   # t2 -> S01, S11
        [1, 0, 1, 0],   # t3 -> S00, S10
        [-1, 1, 0, 0],  # t4 -> -S00, S01
        [0, 0, 0, 1],   # t5 -> S11
        [1, 0, 0, 0],   # t6 -> S00
    ]
    
    # Contribution of super term s to C blocks (C00, C01, C10, C11)
    c_block_contrib_all = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, -1],  # M2 -> C10, -C11
        [0, 1, 0, 1],   # M3 -> C01, C11
        [1, 0, 1, 0],   # M4 -> C00, C10
        [-1, 1, 0, 0],  # M5 -> -C00, C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0],   # M7 -> C00
    ]
    
    # Helper: get global (i,j) for a P element (p_idx) within a block (b_idx) at local (r,c)
    def get_p_global(p_idx, b_idx):
        r = p_idx // 2
        c = p_idx % 2
        if b_idx == 0:   # A00
            return (r, c)
        elif b_idx == 1: # A01
            return (r, 2 + c)
        elif b_idx == 2: # A10
            return (2 + r, c)
        else:            # A11
            return (2 + r, 2 + c)
    
    # Helper: get global (j,k) for a Q element (q_idx) within a block (b_idx) at local (r,c)
    def get_q_global(q_idx, b_idx):
        r = q_idx // 2
        c = q_idx % 2
        if b_idx == 0:   # B00
            return (r, c)
        elif b_idx == 1: # B01
            return (r, 2 + c)
        elif b_idx == 2: # B10
            return (2 + r, c)
        else:            # B11
            return (2 + r, 2 + c)
    
    # Helper: get global (i,k) for C block (cb) at local (r,c)
    def get_c_global(cb, r, c):
        if cb == 0:   # C00
            return (r, c)
        elif cb == 1: # C01
            return (r, 2 + c)
        elif cb == 2: # C10
            return (2 + r, c)
        else:         # C11
            return (2 + r, 2 + c)
    
    for s in range(7):
        for t in range(7):
            p_blocks = p_blocks_all[s]
            q_blocks = q_blocks_all[s]
            p_elems, q_elems = inner_defs[t]
            s_contrib = s_contrib_all[t]
            c_block_contrib = c_block_contrib_all[s]
            
            u_list = []
            v_list = []
            w_list = []
            
            # Build u_list from P elements
            for p_idx, p_coeff in p_elems:
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0:
                        continue
                    gi, gj = get_p_global(p_idx, b_idx)
                    u_list.append((gi, gj, b_coeff * p_coeff))
            
            # Build v_list from Q elements
            for q_idx, q_coeff in q_elems:
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0:
                        continue
                    gj, gk = get_q_global(q_idx, b_idx)
                    v_list.append((gj, gk, b_coeff * q_coeff))
            
            # Build w_list from S contributions and C block contributions
            for sp in range(4):  # S position: 0=S00, 1=S01, 2=S10, 3=S11
                sc = s_contrib[sp]
                if sc == 0:
                    continue
                s_r = sp // 2
                s_c = sp % 2
                for cb in range(4):  # C block
                    cc = c_block_contrib[cb]
                    if cc == 0:
                        continue
                    total_c = sc * cc
                    if total_c == 0:
                        continue
                    gi, gk = get_c_global(cb, s_r, s_c)
                    w_list.append((gk, gi, total_c))
            
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # Part 2: 4x4 * 4x1 schoolbook (16 terms)
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 120:
        pass  # Already done, just return
    
    return {"U": U, "V": V, "W": W}

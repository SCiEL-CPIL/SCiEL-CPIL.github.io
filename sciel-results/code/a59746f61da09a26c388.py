import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 5
    
    # Rank 65: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1)
    rank = 65
    
    # U: (n*m, rank) = (16, 65)
    # V: (m*p, rank) = (20, 65)
    # W: (p*n, rank) = (20, 65)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
        col += 1

    # --- Part 1: Strassen 4x4x4 ---
    
    # Define Super Terms (Block level Strassen)
    # Each super term is defined by coefficients of A blocks and B blocks.
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # Super Term 0: M1 = (A00+A11)(B00+B11)
    # Super Term 1: M2 = (A10+A11)B00
    # Super Term 2: M3 = A00(B01-B11)
    # Super Term 3: M4 = A11(B10-B00)
    # Super Term 4: M5 = (A00+A01)B11
    # Super Term 5: M6 = (A10-A00)(B00+B01)
    # Super Term 6: M7 = (A01-A11)(B10+B11)
    
    super_terms = [
        # (p_blocks, q_blocks, c_block_coeffs)
        # p_blocks: [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
        # q_blocks: [coeff_B00, coeff_B01, coeff_B10, coeff_B11]
        # c_block_coeffs: dict {block_idx: coeff} where block_idx 0:C00, 1:C01, 2:C10, 3:C11
        {
            'p': [1, 0, 0, 1],
            'q': [1, 0, 0, 1],
            'c': {0: 1, 3: 1} # M1 contributes to C00 (+1) and C11 (+1)
        },
        {
            'p': [0, 0, 1, 1],
            'q': [1, 0, 0, 0],
            'c': {2: 1} # M2 contributes to C10 (+1)
        },
        {
            'p': [1, 0, 0, 0],
            'q': [0, 1, 0, -1],
            'c': {1: 1, 3: -1} # M3 contributes to C01 (+1) and C11 (-1)
        },
        {
            'p': [0, 0, 0, 1],
            'q': [-1, 0, 1, 0],
            'c': {0: -1, 2: 1} # M4 contributes to C00 (-1) and C10 (+1)
        },
        {
            'p': [1, 1, 0, 0],
            'q': [0, 0, 0, 1],
            'c': {1: 1} # M5 contributes to C01 (+1)
        },
        {
            'p': [-1, 0, 1, 0],
            'q': [1, 1, 0, 0],
            'c': {0: 1} # M6 contributes to C00 (+1)
        },
        {
            'p': [0, 1, 0, -1],
            'q': [0, 0, 1, 1],
            'c': {3: 1} # M7 contributes to C11 (+1)
        }
    ]
    
    # Inner Strassen Terms (Scalar level Strassen for 2x2 product)
    # For a 2x2 product P*Q, the 7 terms are:
    # 0: (P00+P11)(Q00+Q11)
    # 1: (P10+P11)Q00
    # 2: P00(Q01-Q11)
    # 3: P11(Q10-Q00)
    # 4: (P00+P01)Q11
    # 5: (P10-P00)(Q00+Q01)
    # 6: (P01-P11)(Q10+Q11)
    
    # Contribution of inner term j to R[r,c] (2x2 result)
    # R00 = t0 - t3 + t5
    # R01 = t2 + t4
    # R10 = t1 + t3
    # R11 = t0 - t2 + t6
    
    inner_contribs = [
        # term_idx: { (r,c): coeff }
        { (0,0): 1, (1,1): 1 }, # t0
        { (1,0): 1 },           # t1
        { (0,1): 1, (1,1): -1 },# t2
        { (0,0): -1, (1,0): 1 },# t3
        { (0,1): 1 },           # t4
        { (0,0): 1 },           # t5
        { (1,1): 1 }            # t6
    ]
    
    # Helper to get global indices for A block elements
    def get_a_indices(block_idx, r, c):
        # block_idx: 0:00, 1:01, 2:10, 3:11
        # r, c: local row, col in 2x2 block
        br = block_idx // 2
        bc = block_idx % 2
        gi = 2 * br + r
        gj = 2 * bc + c
        return gi, gj

    def get_b_indices(block_idx, r, c):
        br = block_idx // 2
        bc = block_idx % 2
        gj = 2 * br + r
        gk = 2 * bc + c
        return gj, gk

    for s_idx, st in enumerate(super_terms):
        p_blocks = st['p']
        q_blocks = st['q']
        c_coeffs = st['c']
        
        for i_idx in range(7):
            u_list = []
            v_list = []
            w_list = []
            
            # Determine P and Q element linear combinations for this inner term
            # We need to map inner term i_idx to P and Q elements
            # Inner term 0: (P00+P11)(Q00+Q11)
            # Inner term 1: (P10+P11)Q00
            # Inner term 2: P00(Q01-Q11)
            # Inner term 3: P11(Q10-Q00)
            # Inner term 4: (P00+P01)Q11
            # Inner term 5: (P10-P00)(Q00+Q01)
            # Inner term 6: (P01-P11)(Q10+Q11)
            
            # P elements: 0:P00, 1:P01, 2:P10, 3:P11
            # Q elements: 0:Q00, 1:Q01, 2:Q10, 3:Q11
            
            if i_idx == 0:
                p_elems = {0: 1, 3: 1}
                q_elems = {0: 1, 3: 1}
            elif i_idx == 1:
                p_elems = {2: 1, 3: 1}
                q_elems = {0: 1}
            elif i_idx == 2:
                p_elems = {0: 1}
                q_elems = {1: 1, 3: -1}
            elif i_idx == 3:
                p_elems = {3: 1}
                q_elems = {2: 1, 0: -1}
            elif i_idx == 4:
                p_elems = {0: 1, 1: 1}
                q_elems = {3: 1}
            elif i_idx == 5:
                p_elems = {2: 1, 0: -1}
                q_elems = {0: 1, 1: 1}
            elif i_idx == 6:
                p_elems = {1: 1, 3: -1}
                q_elems = {2: 1, 3: 1}
            
            # Build U list
            for p_elem_idx, p_coeff in p_elems.items():
                pr = p_elem_idx // 2
                pc = p_elem_idx % 2
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0: continue
                    gi, gj = get_a_indices(b_idx, pr, pc)
                    u_list.append((gi, gj, p_coeff * b_coeff))
            
            # Build V list
            for q_elem_idx, q_coeff in q_elems.items():
                qr = q_elem_idx // 2
                qc = q_elem_idx % 2
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0: continue
                    gj, gk = get_b_indices(b_idx, qr, qc)
                    v_list.append((gj, gk, q_coeff * b_coeff))
            
            # Build W list
            # For each C block this super term contributes to
            for c_block_idx, c_block_coeff in c_coeffs.items():
                # c_block_idx: 0:C00, 1:C01, 2:C10, 3:C11
                br = c_block_idx // 2
                bc = c_block_idx % 2
                
                # For each element of R (2x2) that this inner term contributes to
                for (r, c), r_coeff in inner_contribs[i_idx].items():
                    total_coeff = c_block_coeff * r_coeff
                    if total_coeff == 0: continue
                    gi = 2 * br + r
                    gk = 2 * bc + c
                    w_list.append((gk, gi, total_coeff))
            
            add_term(u_list, v_list, w_list)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # --- Part 2: Schoolbook 4x4x1 ---
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(u_list, v_list, w_list)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}

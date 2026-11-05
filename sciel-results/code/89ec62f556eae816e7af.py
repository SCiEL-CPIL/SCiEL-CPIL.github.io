import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Hybrid Strassen: 7 (outer) * 4 (inner schoolbook) = 28 for 4x4x4
    # Schoolbook: 16 for 4x4x1
    rank = 28 + 16
    U = [[0] * rank for _ in range(n * m)]  # 16 x 44
    V = [[0] * rank for _ in range(m * p)]  # 20 x 44
    W = [[0] * rank for _ in range(p * n)]  # 20 x 44
    
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
    
    # Strassen combinations for 4x4 (using 2x2 blocks)
    # P_blocks: coeffs for A00, A01, A10, A11
    # Q_blocks: coeffs for B00, B01, B10, B11
    # C_contrib: (c_block_idx, sign)
    super_terms = [
        # M1: (A00+A11)(B00+B11)
        {'p_blocks': [1, 0, 0, 1], 'q_blocks': [1, 0, 0, 1], 'c_contrib': [(0, 1), (3, 1)]},
        # M2: (A10+A11)(B00)
        {'p_blocks': [0, 0, 1, 1], 'q_blocks': [1, 0, 0, 0], 'c_contrib': [(2, 1), (3, -1)]},
        # M3: (A00)(B01-B11)
        {'p_blocks': [1, 0, 0, 0], 'q_blocks': [0, 1, 0, -1], 'c_contrib': [(1, 1), (3, 1)]},
        # M4: (A11)(B10-B00)
        {'p_blocks': [0, 0, 0, 1], 'q_blocks': [-1, 0, 1, 0], 'c_contrib': [(0, 1), (2, 1)]},
        # M5: (A00+A01)(B11)
        {'p_blocks': [1, 1, 0, 0], 'q_blocks': [0, 0, 0, 1], 'c_contrib': [(0, -1), (1, 1)]},
        # M6: (A10-A00)(B00+B01)
        {'p_blocks': [-1, 0, 1, 0], 'q_blocks': [1, 1, 0, 0], 'c_contrib': [(3, 1)]},
        # M7: (A01-A11)(B10+B11)
        {'p_blocks': [0, 1, 0, -1], 'q_blocks': [0, 0, 1, 1], 'c_contrib': [(0, 1)]},
    ]
    
    # Inner Schoolbook terms for 2x2x2
    # M00 = P00*Q00 + P01*Q10
    # M01 = P00*Q01 + P01*Q11
    # M10 = P10*Q00 + P11*Q10
    # M11 = P10*Q01 + P11*Q11
    
    # p_elem_idx: 0:00, 1:01, 2:10, 3:11
    # q_elem_idx: 0:00, 1:01, 2:10, 3:11
    
    inner_terms = [
        # M00
        [(0, 0), (1, 2)],  # (P00, Q00), (P01, Q10)
        # M01
        [(0, 1), (1, 3)],  # (P00, Q01), (P01, Q11)
        # M10
        [(2, 0), (3, 2)],  # (P10, Q00), (P11, Q10)
        # M11
        [(2, 1), (3, 3)],  # (P10, Q01), (P11, Q11)
    ]
    
    # Helper: get global A indices for P element
    def get_p_indices(p_elem_idx, p_blocks):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0:
                continue
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
    
    # Helper: get global B indices for Q element
    def get_q_indices(q_elem_idx, q_blocks):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0:
                continue
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
    
    # Helper: get global C indices for M element
    # m_elem_idx: 0:00, 1:01, 2:10, 3:11
    def get_c_indices(m_elem_idx, c_contrib):
        r = m_elem_idx // 2
        c = m_elem_idx % 2
        res = []
        for c_block_idx, sign in c_contrib:
            if c_block_idx == 0:  # C00
                gi, gk = r, c
            elif c_block_idx == 1:  # C01
                gi, gk = r, 2 + c
            elif c_block_idx == 2:  # C10
                gi, gk = 2 + r, c
            elif c_block_idx == 3:  # C11
                gi, gk = 2 + r, 2 + c
            # W index is (k, i)
            res.append((gk, gi, sign))
        return res
    
    # Process each super-term
    for super_term in super_terms:
        p_blocks = super_term['p_blocks']
        q_blocks = super_term['q_blocks']
        c_contrib = super_term['c_contrib']
        
        # Process each inner M element (M00, M01, M10, M11)
        for m_idx in range(4):
            # Get the pairs of (P_elem, Q_elem) for this M element
            pairs = inner_terms[m_idx]
            
            for p_elem_idx, q_elem_idx in pairs:
                # Build U list from P element
                u_list = get_p_indices(p_elem_idx, p_blocks)
                
                # Build V list from Q element
                v_list = get_q_indices(q_elem_idx, q_blocks)
                
                # Build W list from C contribution
                # The M element is m_idx
                w_list = get_c_indices(m_idx, c_contrib)
                
                # Merge same indices in u, v, w (though they should be unique per term)
                # Actually, u_list might have multiple entries if p_blocks has multiple non-zero coeffs
                # But for a single P element, it's a linear combo.
                # We just add them all to the same column.
                
                add_term(u_list, v_list, w_list)
    
    # Part 2: 4x4 * 4x1 (rank 16) - schoolbook
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1  # B index: row j, col 4
            W[4 * 4 + i][col] = 1  # C index: k=4, i. W[k*n+i] = W[4*4+i]
            col += 1
    
    return {"U": U, "V": V, "W": W}

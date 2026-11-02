import time
from collections import defaultdict

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    n, m, p = DIMENSIONS
    
    terms = []
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        u = {i: c for i, c in u_coeffs if c != 0}
        v = {i: c for i, c in v_coeffs if c != 0}
        w = {i: c for i, c in w_coeffs if c != 0}
        if u and v and w:
            terms.append((u, v, w))

    # --- Part 1: Strassen for 4x4x4 (B cols 0-3, C cols 0-3) ---
    # A is 4x4. B1 is 4x4 (cols 0-3). C1 is 4x4 (cols 0-3).
    # Split A, B1, C1 into 2x2 blocks.
    
    # Mapping helpers
    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)
        
    def b1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*6 + (y*2+c)
        
    def c1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (y*2+c)*4 + (x*2+r)

    # Strassen 2x2x2 terms (S1-S7)
    def get_strassen_2x2_terms():
        s1 = ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)])
        s2 = ([((1,0), 1), ((1,1), 1)], [((0,0), 1)])
        s3 = ([((0,0), 1)], [((0,1), 1), ((1,1), -1)])
        s4 = ([((1,1), 1)], [((1,0), 1), ((0,0), -1)])
        s5 = ([((0,0), 1), ((0,1), 1)], [((1,1), 1)])
        s6 = ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)])
        s7 = ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
        return [s1, s2, s3, s4, s5, s6, s7]

    strassen_2x2_terms = get_strassen_2x2_terms()
    
    # Result mapping for 2x2 Strassen
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Big Strassen 4x4 combinations
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]), # M1
        ([('10', 1), ('11', 1)], [('00', 1)]),            # M2
        ([('00', 1)], [('01', 1), ('11', -1)]),           # M3
        ([('11', 1)], [('10', 1), ('00', -1)]),           # M4
        ([('00', 1), ('01', 1)], [('11', 1)]),            # M5
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),# M6
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)]) # M7
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
        if time.time() > deadline:
            break
        for s_idx in range(7):
            if time.time() > deadline:
                break
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
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # --- Part 2: Strassen-based 4x4x2 (B cols 4-5, C cols 4-5) ---
    # A is 4x4. B2 is 4x2 (cols 4-5). C2 is 4x2 (cols 4-5).
    # Split A into 2x2 blocks: A00, A01, A10, A11.
    # Split B2 into top 2x2 (T) and bottom 2x2 (B).
    # C2 = [A00*T + A01*B; A10*T + A11*B].
    # Compute 4 products: P1=A00*T, P2=A01*B, P3=A10*T, P4=A11*B.
    # Each product is 2x2x2, rank 7. Total 28 terms.
    
    # Define the 4 products
    products = [
        ('00', 'T', 'top'), # P1 = A00 * T -> C2_top
        ('01', 'B', 'top'), # P2 = A01 * B -> C2_top
        ('10', 'T', 'bot'), # P3 = A10 * T -> C2_bot
        ('11', 'B', 'bot')  # P4 = A11 * B -> C2_bot
    ]
    
    for a_block, b_block, c_part in products:
        if time.time() > deadline:
            break
        for s_idx in range(7):
            if time.time() > deadline:
                break
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            # Construct global U coeffs
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                x, y = int(a_block[0]), int(a_block[1])
                idx = (x*2+rx)*4 + (y*2+cx)
                u_coeffs.append((idx, cx_c))
            
            # Construct global V coeffs
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                if b_block == 'T':
                    idx = ry*6 + (4+cy)
                else: # 'B'
                    idx = (2+ry)*6 + (4+cy)
                v_coeffs.append((idx, cy_c))
            
            # Construct global W coeffs
            w_coeffs = []
            for rc, s_contribs in result_map.items():
                small_coeff = 0
                for s_check, s_c in s_contribs:
                    if s_check == s_idx:
                        small_coeff = s_c
                        break
                if small_coeff == 0:
                    continue
                
                if c_part == 'top':
                    idx = (4+rc[1])*4 + rc[0]
                else: # 'bot'
                    idx = (4+rc[1])*4 + (2+rc[0])
                
                w_coeffs.append((idx, small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    # Merge functions
    def merge_by_uv(terms):
        groups = defaultdict(list)
        for u, v, w in terms:
            key = (tuple(sorted(u.items())), tuple(sorted(v.items())))
            groups[key].append(w)
        merged = []
        for key, w_list in groups.items():
            w_sum = defaultdict(float)
            for w in w_list:
                for i, c in w.items():
                    w_sum[i] += c
            w_sum = {i: c for i, c in w_sum.items() if abs(c) > 1e-9}
            if w_sum:
                u = dict(key[0])
                v = dict(key[1])
                merged.append((u, v, w_sum))
        return merged
    
    def merge_by_uw(terms):
        groups = defaultdict(list)
        for u, v, w in terms:
            key = (tuple(sorted(u.items())), tuple(sorted(w.items())))
            groups[key].append(v)
        merged = []
        for key, v_list in groups.items():
            v_sum = defaultdict(float)
            for v in v_list:
                for i, c in v.items():
                    v_sum[i] += c
            v_sum = {i: c for i, c in v_sum.items() if abs(c) > 1e-9}
            if v_sum:
                u = dict(key[0])
                w = dict(key[1])
                merged.append((u, v_sum, w))
        return merged
    
    def merge_by_vw(terms):
        groups = defaultdict(list)
        for u, v, w in terms:
            key = (tuple(sorted(v.items())), tuple(sorted(w.items())))
            groups[key].append(u)
        merged = []
        for key, u_list in groups.items():
            u_sum = defaultdict(float)
            for u in u_list:
                for i, c in u.items():
                    u_sum[i] += c
            u_sum = {i: c for i, c in u_sum.items() if abs(c) > 1e-9}
            if u_sum:
                v = dict(key[0])
                w = dict(key[1])
                merged.append((u_sum, v, w))
        return merged
    
    # Iterative merging
    current = terms
    for _ in range(10):
        if time.time() > deadline:
            break
        merged_uv = merge_by_uv(current)
        merged_uw = merge_by_uw(current)
        merged_vw = merge_by_vw(current)
        best = min([merged_uv, merged_uw, merged_vw], key=len)
        if len(best) >= len(current):
            break
        current = best
    
    # Convert to matrices
    rank = len(current)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(current):
        for i, c in u.items():
            U[i][col] = c
        for i, c in v.items():
            V[i][col] = c
        for i, c in w.items():
            W[i][col] = c
    
    return {"U": U, "V": V, "W": W}

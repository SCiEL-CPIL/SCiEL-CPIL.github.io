import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
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
    
    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)
        
    def b1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*6 + (y*2+c)
        
    def c1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (y*2+c)*4 + (x*2+r)

    strassen_2x2_terms = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
    ]
    
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]),
        ([('10', 1), ('11', 1)], [('00', 1)]),
        ([('00', 1)], [('01', 1), ('11', -1)]),
        ([('11', 1)], [('10', 1), ('00', -1)]),
        ([('00', 1), ('01', 1)], [('11', 1)]),
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)])
    ]
    
    big_result_map = {
        '00': [(0, 1), (3, 1), (4, -1), (6, 1)],
        '01': [(2, 1), (4, 1)],
        '10': [(1, 1), (3, 1)],
        '11': [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
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
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                b_col = 4 + k
                c_col = 4 + k
                
                u_idx = i * 4 + j
                v_idx = j * 6 + b_col
                w_idx = c_col * 4 + i
                
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
                
    return {"U": U, "V": V, "W": W}

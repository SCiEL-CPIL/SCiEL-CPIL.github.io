import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        add_term(u, v, w)

        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        add_term(u, v, w)

    def strassen_4x4_head():
        # Use the 7-term Strassen decomposition for 4x4x4
        # This is a recursive application: 4x4 = 2x2 blocks of 2x2
        # Each 2x2x2 uses 7 terms, so 4x4x4 uses 7*7=49 terms in naive recursion
        # But we can be smarter
        
        # Define the 7 Strassen terms for 2x2x2
        strassen_2x2 = [
            # (u_indices, v_indices, w_indices) with signs
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        # For 4x4x4, we tile with 2x2 blocks
        # A is 4x4, B is 4x4, C is 4x4
        # Block indices: (br, bc) where br=0,1 and bc=0,1
        for outer_a in range(4):
            for outer_b in range(4):
                for outer_c in range(4):
                    # Determine which 2x2 blocks are involved
                    oa_br, oa_bc = outer_a // 2, outer_a % 2
                    ob_br, ob_bc = outer_b // 2, outer_b % 2
                    oc_br, oc_bc = outer_c // 2, outer_c % 2
                    
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    for inner in range(7):
                        for (ir, ic, coeff_a) in strassen_2x2[inner][0]:
                            # Map to actual A indices
                            a_row = oa_br * 2 + ir
                            a_col = oa_bc * 2 + ic
                            u[a_row * m + a_col] += coeff_a
                        
                        for (jr, jc, coeff_b) in strassen_2x2[inner][1]:
                            # Map to actual B indices  
                            b_row = ob_br * 2 + jr
                            b_col = ob_bc * 2 + jc
                            v[b_row * p + b_col] += coeff_b
                        
                        for (kr, kc, coeff_c) in strassen_2x2[inner][2]:
                            # Map to actual C indices (transposed convention)
                            c_row = oc_br * 2 + kr
                            c_col = oc_bc * 2 + kc
                            w[c_row * n + c_col] += coeff_c
                    
                    add_term(u, v, w)

    def strassen_4x4x2_tail():
        # For the remaining 2 columns of B (columns 4,5)
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                strassen_2x2_block(i_off, j_off, k_off)

    def schoolbook_4x4x1_tail():
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                add_term(u, v, w)

    strassen_4x4_head()
    strassen_4x4x2_tail()
    schoolbook_4x4x1_tail()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}

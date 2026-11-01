def construct():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (indices 0,1,2,3)
    # u1: a+d
    # u2: c+d
    # u3: a
    # u4: d
    # u5: a+b
    # u6: c-a
    # u7: b-d
    u_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # For each block r (0..3)
    for r in range(4):
        # B indices for this block
        # B_{0, 2r} -> 2r
        # B_{0, 2r+1} -> 2r+1
        # B_{1, 2r} -> 8+2r
        # B_{1, 2r+1} -> 8+2r+1
        b_idx_00 = 2 * r
        b_idx_01 = 2 * r + 1
        b_idx_10 = 8 + 2 * r
        b_idx_11 = 8 + 2 * r + 1
        
        # C indices for this block
        # C_{0, 2r} -> 4r
        # C_{0, 2r+1} -> 4r+2
        # C_{1, 2r} -> 4r+1
        # C_{1, 2r+1} -> 4r+3
        c_idx_00 = 4 * r
        c_idx_01 = 4 * r + 2
        c_idx_10 = 4 * r + 1
        c_idx_11 = 4 * r + 3
        
        # v forms and w forms for the 7 products
        # M1: (a+d)(e+h) -> v: e+h, w: C00+C11
        # M2: (c+d)e -> v: e, w: C10-C11
        # M3: a(f-h) -> v: f-h, w: C01+C11
        # M4: d(g-e) -> v: g-e, w: C00+C10
        # M5: (a+b)h -> v: h, w: -C00+C01
        # M6: (c-a)(e+f) -> v: e+f, w: C11
        # M7: (b-d)(g+h) -> v: g+h, w: C00
        
        # Note: w forms are coefficients for C entries.
        # w vector has length 16.
        
        v_forms = [
            {b_idx_00: 1, b_idx_10: 1},       # M1: e+h
            {b_idx_00: 1},                     # M2: e
            {b_idx_01: 1, b_idx_11: -1},      # M3: f-h
            {b_idx_10: 1, b_idx_00: -1},      # M4: g-e
            {b_idx_11: 1},                     # M5: h
            {b_idx_00: 1, b_idx_01: 1},       # M6: e+f
            {b_idx_10: 1, b_idx_11: 1}        # M7: g+h
        ]
        
        w_forms = [
            {c_idx_00: 1, c_idx_11: 1},       # M1
            {c_idx_10: 1, c_idx_11: -1},      # M2
            {c_idx_01: 1, c_idx_11: 1},       # M3
            {c_idx_00: 1, c_idx_10: 1},       # M4
            {c_idx_00: -1, c_idx_01: 1},      # M5
            {c_idx_11: 1},                    # M6
            {c_idx_00: 1}                     # M7
        ]
        
        for s in range(7):
            col = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][col] = u_forms[s][i]
                
            # Fill V
            for idx, val in v_forms[s].items():
                V[idx][col] = val
                
            # Fill W
            for idx, val in w_forms[s].items():
                W[idx][col] = val
                
    return {"U": U, "V": V, "W": W}

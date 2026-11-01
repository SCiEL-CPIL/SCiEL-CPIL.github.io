def construct():
    n, m, p = 2, 2, 7
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U: [A00, A01, A10, A11]
    # V: [B00, B01, B10, B11]
    # W: {C00: coeff, C01: coeff, C10: coeff, C11: coeff}
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], {0: 1, 3: 1}),       # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], {2: 1}),             # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], {1: 1, 3: 1}),      # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], {0: 1, 2: 1}),      # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], {0: -1, 1: 1}),      # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], {3: 1}),            # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], {0: 1}),            # M7
    ]
    
    r = 0
    # Process 3 pairs of columns: (0,1), (2,3), (4,5)
    for k in range(3):
        col_start = 2 * k
        # Global indices for V
        v_00 = 0 * p + col_start       # B[0, col_start]
        v_01 = 0 * p + (col_start + 1) # B[0, col_start+1]
        v_10 = 1 * p + col_start       # B[1, col_start]
        v_11 = 1 * p + (col_start + 1) # B[1, col_start+1]
        
        # Global indices for W
        # C is 2x7. W index is k*n + i.
        # C[0, col_start] -> (col_start)*2 + 0
        # C[0, col_start+1] -> (col_start+1)*2 + 0
        # C[1, col_start] -> (col_start)*2 + 1
        # C[1, col_start+1] -> (col_start+1)*2 + 1
        w_00 = col_start * n + 0
        w_01 = (col_start + 1) * n + 0
        w_10 = col_start * n + 1
        w_11 = (col_start + 1) * n + 1
        
        w_map = {
            0: w_00,
            1: w_01,
            2: w_10,
            3: w_11
        }
        
        for u_vec, v_vec, w_coeff in strassen_terms:
            for i in range(4):
                if u_vec[i] != 0:
                    U[i][r] = u_vec[i]
            
            v_indices = [v_00, v_01, v_10, v_11]
            for i in range(4):
                if v_vec[i] != 0:
                    V[v_indices[i]][r] = v_vec[i]
            
            for local_w_idx, coeff in w_coeff.items():
                global_w_idx = w_map[local_w_idx]
                W[global_w_idx][r] = coeff
                
            r += 1
            
    # Process singleton column 6
    col_single = 6
    v_0 = 0 * p + col_single
    v_1 = 1 * p + col_single
    
    w_0 = col_single * n + 0
    w_1 = col_single * n + 1
    
    singleton_terms = [
        ([1, 0, 0, 0], [1, 0], {0: 1}), # A00 * B06 -> C06
        ([0, 1, 0, 0], [0, 1], {0: 1}), # A01 * B16 -> C06
        ([0, 0, 1, 0], [1, 0], {1: 1}), # A10 * B06 -> C16
        ([0, 0, 0, 1], [0, 1], {1: 1}), # A11 * B16 -> C16
    ]
    
    for u_vec, v_vec, w_coeff in singleton_terms:
        for i in range(4):
            if u_vec[i] != 0:
                U[i][r] = u_vec[i]
        
        v_indices = [v_0, v_1]
        for i in range(2):
            if v_vec[i] != 0:
                V[v_indices[i]][r] = v_vec[i]
        
        w_map = {0: w_0, 1: w_1}
        for local_w_idx, coeff in w_coeff.items():
            global_w_idx = w_map[local_w_idx]
            W[global_w_idx][r] = coeff
            
        r += 1
        
    return {"U": U, "V": V, "W": W}

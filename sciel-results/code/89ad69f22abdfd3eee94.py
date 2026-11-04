import time

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    The decomposition is based on applying Strassen's algorithm (rank 7) to four 
    disjoint 2x2x2 blocks. The disjointness of the input supports for each block 
    ensures that the ranks add up, giving a total rank of 4 * 7 = 28.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 4
    rank = 28
    
    # Strassen patterns for 2x2 multiplication
    # Each pattern is (u_dict, v_dict, w_dict)
    # Indices 0..3 correspond to 2x2 matrix elements (row-major: 00, 01, 10, 11)
    strassen_patterns = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),      # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),      # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),      # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),      # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),      # M7
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 4 blocks
    # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
    blocks = [
        (0, 0, 0, 0),   # A_L B_TL -> C_L
        (2, 2, 0, 0),   # A_R B_BL -> C_L
        (0, 0, 2, 2),   # A_L B_TR -> C_R
        (2, 2, 2, 2),   # A_R B_BR -> C_R
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
                
            # Fill U
            for local_idx, val in u_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = local_r * m + (a_off + local_c)
                U[g_idx][col_idx] = val
            
            # Fill V
            for local_idx, val in v_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = (b_row_off + local_r) * p + (b_col_off + local_c)
                V[g_idx][col_idx] = val
            
            # Fill W
            for local_idx, val in w_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_i = local_r
                g_k = c_off + local_c
                g_idx = g_k * n + g_i
                W[g_idx][col_idx] = val
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}

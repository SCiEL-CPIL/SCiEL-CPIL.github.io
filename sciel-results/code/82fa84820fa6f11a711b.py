import time

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    # Strassen patterns for 2x2 multiplication
    strassen_patterns = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
            for local_idx, val in u_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = local_r * m + (a_off + local_c)
                U[g_idx][col_idx] = val
            
            for local_idx, val in v_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_idx = (b_row_off + local_r) * p + (b_col_off + local_c)
                V[g_idx][col_idx] = val
            
            for local_idx, val in w_coeffs.items():
                local_r = local_idx // 2
                local_c = local_idx % 2
                g_i = local_r
                g_k = c_off + local_c
                g_idx = g_k * n + g_i
                W[g_idx][col_idx] = val
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}

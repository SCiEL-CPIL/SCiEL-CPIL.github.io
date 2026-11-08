import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # Local indices: 0=00, 1=01, 2=10, 3=11
    s_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]

    def add_term(u_indices, v_indices, w_indices, col_idx):
        for idx, coeff in u_indices:
            U[idx, col_idx] += coeff
        for idx, coeff in v_indices:
            V[idx, col_idx] += coeff
        for idx, coeff in w_indices:
            W[idx, col_idx] += coeff

    def apply_block(a_off, b_row_off, b_col_off, c_col_off, start_col):
        for i, (u_loc, v_loc, w_loc) in enumerate(s_terms):
            col = start_col + i
            
            u_global = []
            for loc_idx, coeff in u_loc:
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_off * 2 + c
                g_idx = g_r * 4 + g_c
                u_global.append((g_idx, coeff))
            
            v_global = []
            for loc_idx, coeff in v_loc:
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_off * 2 + r
                g_c = b_col_off * 2 + c
                g_idx = g_r * 4 + g_c
                v_global.append((g_idx, coeff))
            
            w_global = []
            for loc_idx, coeff in w_loc:
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_off * 2 + c
                # W[k*n + i] -> k is col, i is row
                g_idx = g_c * 2 + g_r
                w_global.append((g_idx, coeff))
                
            add_term(u_global, v_global, w_global, col)

    # 1. A1 * BTL -> CL
    apply_block(0, 0, 0, 0, 0)
    # 2. A2 * BBL -> CL
    apply_block(1, 1, 0, 0, 7)
    # 3. A1 * BTR -> CR
    apply_block(0, 0, 1, 1, 14)
    # 4. A2 * BBR -> CR
    apply_block(1, 1, 1, 1, 21)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

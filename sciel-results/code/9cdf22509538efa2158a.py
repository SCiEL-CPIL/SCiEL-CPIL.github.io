"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define base 2x2 Strassen factors
    # Indices: 0:00, 1:01, 2:10, 3:11
    base_factors = []
    
    # M1
    base_factors.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}))
    # M2
    base_factors.append(({2:1, 3:1}, {0:1}, {2:1, 3:-1}))
    # M3
    base_factors.append(({0:1}, {2:1, 3:-1}, {1:1, 3:1}))
    # M4
    base_factors.append(({3:1}, {1:1, 0:-1}, {0:1, 2:1}))
    # M5
    base_factors.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
    # M6
    base_factors.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))
    # M7
    base_factors.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))
    
    # Top-level Strassen patterns for 4x4 (blocks of 2x2)
    top_patterns = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {2:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {1:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    # Generate 49 columns for the 4x4x4 part
    for pat_idx, (a_coefs, b_coefs, c_coefs) in enumerate(top_patterns):
        for base_idx, (u_base, v_base, w_base) in enumerate(base_factors):
            u_col = {}
            v_col = {}
            w_col = {}
            
            # Process U
            for virt_idx, val in u_base.items():
                r_v = virt_idx // 2
                c_v = virt_idx % 2
                for blk_idx, coeff in a_coefs.items():
                    r_b = blk_idx // 2
                    c_b = blk_idx % 2
                    phys_row = (r_b * 2 + r_v) * 2 + (c_b * 2 + c_v)
                    u_col[phys_row] = u_col.get(phys_row, 0) + val * coeff
            
            # Process V
            for virt_idx, val in v_base.items():
                r_v = virt_idx // 2
                c_v = virt_idx % 2
                for blk_idx, coeff in b_coefs.items():
                    r_b = blk_idx // 2
                    c_b = blk_idx % 2
                    phys_row = (r_b * 2 + r_v) * 2 + (c_b * 2 + c_v)
                    v_col[phys_row] = v_col.get(phys_row, 0) + val * coeff
            
            # Process W
            for virt_idx, val in w_base.items():
                r_v = virt_idx // 2
                c_v = virt_idx % 2
                for blk_idx, coeff in c_coefs.items():
                    r_b = blk_idx // 2
                    c_b = blk_idx % 2
                    phys_row = (r_b * 2 + r_v) * 2 + (c_b * 2 + c_v)
                    w_col[phys_row] = w_col.get(phys_row, 0) + val * coeff
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
            
    # Add 4x1 part (16 columns)
    for i in range(4):
        for j in range(4):
            u_col = {i * 4 + j: 1}
            v_col = {j * 5 + 4: 1}
            w_col = {4 * 4 + i: 1}
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
            
    rank = len(U_list)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r, val in U_list[c].items():
            U[r][c] = val
        for r, val in V_list[c].items():
            V[r][c] = val
        for r, val in W_list[c].items():
            W[r][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

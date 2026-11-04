def construct():
    n, m, p = 4, 4, 9
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2x2 base case
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def add_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Add a 4x4x4 Strassen decomposition using 2x2 blocks."""
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                # Fill u_vec from A_blocks
                for (loc_i, loc_j), coeff in u_c.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                # Fill v_vec from B_blocks
                for (loc_j, loc_k), coeff in v_c.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                # Fill w_vec from C_blocks
                for (loc_i, loc_k), coeff in w_c.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # 7 Strassen combinations for 4x4 using 2x2 blocks
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])

    def add_schoolbook_block(B_col_start, num_cols):
        """Add schoolbook terms for a 4x4 x 4xk block."""
        for k in range(num_cols):
            global_k = B_col_start + k
            for i in range(n):
                for j in range(m):
                    u_vec = [0] * (n*m)
                    v_vec = [0] * (m*p)
                    w_vec = [0] * (p*n)
                    u_vec[i * m + j] = 1
                    v_vec[j * p + global_k] = 1
                    w_vec[global_k * n + i] = 1
                    add_term(u_vec, v_vec, w_vec)

    # Block 1: B columns 0-2 (4x3), Schoolbook
    add_schoolbook_block(0, 3)
    
    # Block 2: B columns 3-6 (4x4), Strassen
    add_strassen_4x4(0, 0, 0, 3, 0, 3)
    
    # Block 3: B columns 7-8 (4x2), Schoolbook
    add_schoolbook_block(7, 2)
    
    rank = len(U_cols)
    
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n*m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m*p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p*n)]
    
    return {"U": U, "V": V, "W": W}

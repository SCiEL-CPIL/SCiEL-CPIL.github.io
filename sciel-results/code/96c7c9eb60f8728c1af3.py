    """Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
    DIMENSIONS = [6, 6, 6]

    # EVOLVE-BLOCK-START
    def construct():
        n, m, p = DIMENSIONS
        rank = 189
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for 2x2 block matrix multiplication
        # Each entry is a list of (block_row, block_col, coeff)
        
        cA = [
            [(0,0,1), (1,1,1)],      # M1: A00 + A11
            [(1,0,1), (1,1,1)],      # M2: A10 + A11
            [(0,0,1)],               # M3: A00
            [(1,1,1)],               # M4: A11
            [(0,0,1), (0,1,1)],      # M5: A00 + A01
            [(1,0,1), (0,0,-1)],     # M6: A10 - A00
            [(0,1,1), (1,1,-1)]      # M7: A01 - A11
        ]
        
        cB = [
            [(0,0,1), (1,1,1)],      # M1: B00 + B11
            [(0,0,1)],               # M2: B00
            [(0,1,1), (1,1,-1)],     # M3: B01 - B11
            [(1,0,1), (0,0,-1)],     # M4: B10 - B00
            [(1,1,1)],               # M5: B11
            [(0,0,1), (0,1,1)],      # M6: B00 + B01
            [(1,0,1), (1,1,1)]       # M7: B10 + B11
        ]
        
        d = [
            [(0,0,1), (1,1,1)],      # M1 -> C00, C11
            [(1,0,1), (1,1,-1)],     # M2 -> C10, C11
            [(0,1,1), (1,1,1)],      # M3 -> C01, C11
            [(0,0,1), (1,0,1)],      # M4 -> C00, C10
            [(0,0,-1), (0,1,1)],     # M5 -> C00, C01
            [(1,1,1)],               # M6 -> C11
            [(0,0,1)]                # M7 -> C00
        ]
        
        for s in range(7):
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        c_idx = s * 27 + i * 9 + k * 3 + j
                        
                        # Update U (A input)
                        # U index: i_global * m + j_global
                        for ar, ac, val in cA[s]:
                            u_row = 3 * ar + i
                            u_col = 3 * ac + k
                            u_idx = u_row * 6 + u_col
                            U[u_idx][c_idx] = val
                            
                        # Update V (B input)
                        # V index: j_global * p + k_global
                        for br, bc, val in cB[s]:
                            v_row = 3 * br + k
                            v_col = 3 * bc + j
                            v_idx = v_row * 6 + v_col
                            V[v_idx][c_idx] = val
                            
                        # Update W (C output)
                        # W index: k_global * n + i_global (Column-major)
                        # C block br, bc at local i, j
                        # Global C row = 3*br + i, Global C col = 3*bc + j
                        for cr, cc, val in d[s]:
                            w_col = 3 * cc + j
                            w_row = 3 * cr + i
                            w_idx = w_col * 6 + w_row
                            W[w_idx][c_idx] = val

        return {"U": U, "V": V, "W": W}
    # EVOLVE-BLOCK-END
    
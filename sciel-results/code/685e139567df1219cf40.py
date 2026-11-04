"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen 2x2 on 3x3 blocks with Schoolbook 3x3 base.
    # Rank = 7 * 9 = 63
    
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A, B, C
    # alpha[k, b_i, b_j]
    alpha = {}
    # beta[k, b_j, b_k]
    beta = {}
    # gamma[k, b_k, b_i]
    gamma = {}
    
    # A_k definitions
    # k=0: A00 + A11
    alpha[(0, 0, 0)] = 1; alpha[(0, 1, 1)] = 1
    # k=1: A10 + A11
    alpha[(1, 1, 0)] = 1; alpha[(1, 1, 1)] = 1
    # k=2: A00
    alpha[(2, 0, 0)] = 1
    # k=3: A11
    alpha[(3, 1, 1)] = 1
    # k=4: A00 + A01
    alpha[(4, 0, 0)] = 1; alpha[(4, 0, 1)] = 1
    # k=5: A10 - A00
    alpha[(5, 1, 0)] = 1; alpha[(5, 0, 0)] = -1
    # k=6: A01 - A11
    alpha[(6, 0, 1)] = 1; alpha[(6, 1, 1)] = -1
    
    # B_k definitions
    # k=0: B00 + B11
    beta[(0, 0, 0)] = 1; beta[(0, 1, 1)] = 1
    # k=1: B00
    beta[(1, 0, 0)] = 1
    # k=2: B11 - B00
    beta[(2, 1, 1)] = 1; beta[(2, 0, 0)] = -1
    # k=3: B10 - B00
    beta[(3, 1, 0)] = 1; beta[(3, 0, 0)] = -1
    # k=4: B11
    beta[(4, 1, 1)] = 1
    # k=5: B00 + B01
    beta[(5, 0, 0)] = 1; beta[(5, 0, 1)] = 1
    # k=6: B10 + B11
    beta[(6, 1, 0)] = 1; beta[(6, 1, 1)] = 1
    
    # C output reconstruction
    # C00 = M1 + M4 - M5 + M7
    gamma[(0, 0, 0)] = 1
    gamma[(3, 0, 0)] = 1
    gamma[(4, 0, 0)] = -1
    gamma[(6, 0, 0)] = 1
    # C01 = M3 + M5
    gamma[(2, 1, 0)] = 1
    gamma[(4, 1, 0)] = 1
    # C10 = M2 + M4
    gamma[(1, 0, 1)] = 1
    gamma[(3, 0, 1)] = 1
    # C11 = M1 - M2 + M3 + M6
    gamma[(0, 1, 1)] = 1
    gamma[(1, 1, 1)] = -1
    gamma[(2, 1, 1)] = 1
    gamma[(5, 1, 1)] = 1

    # Iterate over all columns (k_str, c_win)
    for k_str in range(7):
        for c_win in range(9):
            col_idx = k_str * 9 + c_win
            
            # c_win corresponds to s_i * 3 + s_j
            s_i = c_win // 3
            s_j = c_win % 3
            
            # Fill U
            # U row u = i * 6 + j = (b_i * 3 + s_i) * 6 + (b_j * 3 + s_j)
            for b_i in range(2):
                for b_j in range(2):
                    u = (b_i * 3 + s_i) * 6 + (b_j * 3 + s_j)
                    coeff = alpha.get((k_str, b_i, b_j), 0)
                    U[u][col_idx] = coeff
            
            # Fill V
            # V row v = j * 6 + k = (b_j * 3 + s_j) * 6 + (b_k * 3 + s_k)
            # c_win fixes s_j. s_k is free in the row index, but V_3 logic says:
            # V_3 row is s_j * 3 + s_k.
            # V_3[s_j * 3 + s_k, c_win] = 1 if c_win == s_i * 3 + s_j.
            # So for any s_k, if s_j matches, V_3 has 1.
            # Thus for any b_k, s_k, if s_j matches, V[v, col] = beta.
            for b_j in range(2):
                if True: # s_j is fixed by c_win, so it always matches the requirement for V_3
                    for b_k in range(2):
                        for s_k in range(3):
                            v = (b_j * 3 + s_j) * 6 + (b_k * 3 + s_k)
                            coeff = beta.get((k_str, b_j, b_k), 0)
                            V[v][col_idx] = coeff

            # Fill W
            # W row w = k * 6 + i = (b_k * 3 + s_k) * 6 + (b_i * 3 + s_i)
            # W_3 row is s_k * 3 + s_i.
            # W_3[s_k * 3 + s_i, c_win] = 1 if c_win == s_i * 3 + s_j.
            # So for any s_k, if s_i matches, W_3 has 1.
            for b_k in range(2):
                for s_k in range(3):
                    for b_i in range(2):
                        w = (b_k * 3 + s_k) * 6 + (b_i * 3 + s_i)
                        coeff = gamma.get((k_str, b_k, b_i), 0)
                        W[w][col_idx] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

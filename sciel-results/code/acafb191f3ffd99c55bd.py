"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define Strassen coefficients for 2x2 blocks
    # alpha[k_out][I][J], beta[k_out][J][K], gamma[k_out][I][K]
    # k_out is 1-indexed in logic, 0-indexed in lists
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # alpha[k][I][J]
    alpha = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    
    # beta[k][J][K]
    beta = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    
    # gamma[k][I][K]
    gamma = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]

    # Inner Strassen coefficients are identical in structure
    # x[s][i][j], y[s][j][k], w[s][i][k]
    # We can reuse the same arrays
    
    for k_out in range(7):
        for s_in in range(7):
            r = k_out * 7 + s_in
            
            # Loop over block indices I, J, K
            for I in range(2):
                for J in range(2):
                    a_coeff = alpha[k_out][I][J]
                    if a_coeff == 0:
                        continue
                    for K in range(2):
                        b_coeff = beta[k_out][J][K]
                        if b_coeff == 0:
                            continue
                        g_coeff = gamma[k_out][I][K]
                        if g_coeff == 0:
                            continue
                        
                        # Loop over element indices i, j, k
                        for i in range(2):
                            for j in range(2):
                                x_coeff = alpha[s_in][i][j]
                                if x_coeff == 0:
                                    continue
                                u_val = a_coeff * x_coeff
                                if u_val == 0:
                                    continue
                                u_idx = (2 * I + i) * 4 + (2 * J + j)
                                U[u_idx][r] += u_val
                                
                                for k in range(2):
                                    y_coeff = beta[s_in][j][k]
                                    if y_coeff == 0:
                                        continue
                                    w_coeff = gamma[s_in][i][k]
                                    if w_coeff == 0:
                                        continue
                                    
                                    v_val = b_coeff * y_coeff
                                    w_val = g_coeff * w_coeff
                                    
                                    if v_val != 0:
                                        v_idx = (2 * J + j) * 4 + (2 * K + k)
                                        V[v_idx][r] += v_val
                                        
                                    if w_val != 0:
                                        w_idx = (2 * K + k) * 4 + (2 * I + i)
                                        W[w_idx][r] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49 * 2 = 98
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 multiplication
    # alpha[s][i][j], beta[s][j][k], gamma[s][i][k]
    alpha = [[0]*2 for _ in range(2)] # placeholder
    alpha = [[[0]*2 for _ in range(2)] for _ in range(7)]
    beta = [[[0]*2 for _ in range(2)] for _ in range(7)]
    gamma = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # M0
    alpha[0][0][0] = 1; alpha[0][1][1] = 1
    beta[0][0][0] = 1; beta[0][1][1] = 1
    # M1
    alpha[1][1][0] = 1; alpha[1][1][1] = 1
    beta[1][0][0] = 1
    # M2
    alpha[2][0][0] = 1
    beta[2][0][1] = 1; beta[2][1][1] = -1
    # M3
    alpha[3][1][1] = 1
    beta[3][1][0] = 1; beta[3][0][0] = -1
    # M4
    alpha[4][0][0] = 1; alpha[4][0][1] = 1
    beta[4][1][1] = 1
    # M5
    alpha[5][1][0] = 1; alpha[5][0][0] = -1
    beta[5][0][0] = 1; beta[5][0][1] = 1
    # M6
    alpha[6][0][1] = 1; alpha[6][1][1] = -1
    beta[6][1][0] = 1; beta[6][1][1] = 1
    
    # Gamma (C = sum gamma * M)
    # C00 = M0 - M3 + M6
    gamma[0][0][0] = 1; gamma[3][0][0] = -1; gamma[6][0][0] = 1
    # C01 = M2 + M4
    gamma[2][0][1] = 1; gamma[4][0][1] = 1
    # C10 = M1 + M3
    gamma[1][1][0] = 1; gamma[3][1][0] = 1
    # C11 = M0 - M2 + M5
    gamma[0][1][1] = 1; gamma[2][1][1] = -1; gamma[5][1][1] = 1
    
    # Define Outer Strassen coefficients for 4x4 (block level)
    # x_s[b_i][b_j], y_s[b_i][b_j]
    # b_i, b_j in {0, 1}
    x = [[[0]*2 for _ in range(2)] for _ in range(7)]
    y = [[[0]*2 for _ in range(2)] for _ in range(7)]
    gamma_outer = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # M0 = (A00+A11)(B00+B11)
    x[0][0][0] = 1; x[0][1][1] = 1
    y[0][0][0] = 1; y[0][1][1] = 1
    # M1 = (A10+A11)B00
    x[1][1][0] = 1; x[1][1][1] = 1
    y[1][0][0] = 1
    # M2 = A00(B01-B11)
    x[2][0][0] = 1
    y[2][0][1] = 1; y[2][1][1] = -1
    # M3 = A11(B10-B00)
    x[3][1][1] = 1
    y[3][1][0] = 1; y[3][0][0] = -1
    # M4 = (A00+A01)B11
    x[4][0][0] = 1; x[4][0][1] = 1
    y[4][1][1] = 1
    # M5 = (A10-A00)(B00+B01)
    x[5][1][0] = 1; x[5][0][0] = -1
    y[5][0][0] = 1; y[5][0][1] = 1
    # M6 = (A01-A11)(B10+B11)
    x[6][0][1] = 1; x[6][1][1] = -1
    y[6][1][0] = 1; y[6][1][1] = 1
    
    # Gamma Outer (C_block = sum gamma_outer * M_outer)
    # C00 = M0 - M3 + M6
    gamma_outer[0][0][0] = 1; gamma_outer[3][0][0] = -1; gamma_outer[6][0][0] = 1
    # C01 = M2 + M4
    gamma_outer[2][0][1] = 1; gamma_outer[4][0][1] = 1
    # C10 = M1 + M3
    gamma_outer[1][1][0] = 1; gamma_outer[3][1][0] = 1
    # C11 = M0 - M2 + M5
    gamma_outer[0][1][1] = 1; gamma_outer[2][1][1] = -1; gamma_outer[5][1][1] = 1
    
    # Generate terms
    col = 0
    for s in range(7):
        for t in range(7):
            # Calculate U, V, W for this term
            # U[i,j] = sum over blocks and local indices of alpha_inner[t][li][lj] * x[s][bi][bj]
            for i in range(4):
                for j in range(4):
                    i_block = i // 2
                    i_local = i % 2
                    j_block = j // 2
                    j_local = j % 2
                    
                    coeff_u = 0
                    for bi in range(2):
                        for bj in range(2):
                            if i_block == bi and j_block == bj:
                                coeff_u += alpha[t][i_local][j_local] * x[s][bi][bj]
                    U[i * m + j][col] = coeff_u
            
            for j in range(4):
                for k in range(4):
                    j_block = j // 2
                    j_local = j % 2
                    k_block = k // 2
                    k_local = k % 2
                    
                    coeff_v = 0
                    for bi in range(2):
                        for bk in range(2):
                            if j_block == bi and k_block == bk:
                                coeff_v += beta[t][j_local][k_local] * y[s][bi][bk]
                    V[j * p + k][col] = coeff_v
            
            for i in range(4):
                for k in range(4):
                    i_block = i // 2
                    i_local = i % 2
                    k_block = k // 2
                    k_local = k % 2
                    
                    coeff_w = gamma_outer[s][i_block][k_block] * gamma[t][i_local][k_local]
                    W[k * n + i][col] = coeff_w
            
            # Second half (B_R)
            # V index shift: k -> k + 4
            # W index shift: k -> k + 4
            for j in range(4):
                for k in range(4):
                    V[j * p + k][col + 49] = V[j * p + k][col]
            
            for i in range(4):
                for k in range(4):
                    W[(k + 4) * n + i][col + 49] = W[k * n + i][col]
            
            # U is the same, so copy
            for i in range(4):
                for j in range(4):
                    U[i * m + j][col + 49] = U[i * m + j][col]
            
            col += 1
            if col == 49:
                break
        if col == 49:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

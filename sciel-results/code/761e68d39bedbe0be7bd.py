import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A blocks: 2x2, each 1x2
    # B blocks: 2x2, each 2x4
    # C blocks: 2x2, each 1x4
    
    # alpha[s][i][j] : coeff for A[i][j] in M_s
    # beta[s][k][l]  : coeff for B[k][l] in M_s
    # gamma[s][i][l] : coeff for M_s in C[i][l]
    
    # s=1: (A00+A11)(B00+B11)
    alpha_1 = [[1, 0], [0, 1]]
    beta_1  = [[1, 0], [0, 1]]
    gamma_1 = [[1, 0], [0, 1]] # C00 += M1, C11 += M1
    
    # s=2: (A10+A11)B00
    alpha_2 = [[0, 0], [1, 1]]
    beta_2  = [[1, 0], [0, 0]]
    gamma_2 = [[0, 0], [1, 0]] # C10 += M2
    
    # s=3: A00(B10-B00)
    alpha_3 = [[1, 0], [0, 0]]
    beta_3  = [[-1, 0], [1, 0]]
    gamma_3 = [[0, 1], [0, 0]] # C01 += M3, C11 += M3
    
    # s=4: A11(B11-B01)
    alpha_4 = [[0, 0], [0, 1]]
    beta_4  = [[0, -1], [0, 1]]
    gamma_4 = [[1, 0], [0, 1]] # C00 += M4, C11 += M4
    
    # s=5: (A00+A01)B11
    alpha_5 = [[1, 1], [0, 0]]
    beta_5  = [[0, 0], [0, 1]]
    gamma_5 = [[-1, 0], [0, 1]] # C00 -= M5, C11 -= M5? 
    # Strassen: C00 = M1+M4-M5+M7, C11 = M1-M2+M3+M6
    # So gamma_5[0][0] = -1, gamma_5[1][1] = 0? 
    # Wait, standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M5 appears in C00 (-1) and C01 (+1).
    gamma_5 = [[-1, 1], [0, 0]]
    
    # s=6: (A10-A00)(B00+B01)
    alpha_6 = [[-1, 0], [1, 0]]
    beta_6  = [[1, 1], [0, 0]]
    gamma_6 = [[0, 0], [0, 1]] # C11 += M6
    
    # s=7: (A01-A11)(B10+B11)
    alpha_7 = [[0, 1], [0, -1]]
    beta_7  = [[0, 1], [1, 1]]
    gamma_7 = [[1, 0], [0, 0]] # C00 += M7
    
    alphas = [None, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7]
    betas  = [None, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7]
    gammas = [None, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6, gamma_7]
    
    for s in range(1, 8):
        for p_idx in range(2): # p in {0, 1}
            for q_idx in range(4): # q in {0, 1, 2, 3}
                r = (s - 1) * 8 + p_idx * 4 + q_idx
                
                # Update U
                for i_prime in range(2):
                    for j_prime in range(2):
                        a_val = alphas[s][i_prime][j_prime]
                        if a_val != 0:
                            # A index: i_prime * m + (2*j_prime + p_idx)
                            u_idx = i_prime * m + (2 * j_prime + p_idx)
                            U[u_idx][r] = a_val
                
                # Update V
                for k_prime in range(2):
                    for l_prime in range(2):
                        b_val = betas[s][k_prime][l_prime]
                        if b_val != 0:
                            # B index: (2*k_prime + p_idx) * p + (4*l_prime + q_idx)
                            v_idx = (2 * k_prime + p_idx) * p + (4 * l_prime + q_idx)
                            V[v_idx][r] = b_val
                
                # Update W
                for i_out in range(2):
                    for l_out in range(2):
                        g_val = gammas[s][i_out][l_out]
                        if g_val != 0:
                            # W index: (4*l_out + q_idx) * n + i_out
                            w_idx = (4 * l_out + q_idx) * n + i_out
                            W[w_idx][r] = g_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

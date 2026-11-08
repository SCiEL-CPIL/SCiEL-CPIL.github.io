import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use block size 2, so 3x3 blocks.
    block_size = 2
    num_blocks = n // block_size
    
    rank = (num_blocks ** 2) * 7 # 9 * 7 = 63
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products coefficients
    # M_r = (sum alpha_{r, u', v'} A_{u', v'}) * (sum beta_{r, v'', w'} B_{v'', w'})
    # A is 2x2, B is 2x2.
    # Indices: A[u', v'], B[v'', w']
    
    # r=0: (A00 + A11) * (B00 + B11)
    alpha_0 = {(0,0): 1, (1,1): 1}
    beta_0 = {(0,0): 1, (1,1): 1}
    
    # r=1: (A10 + A11) * B00
    alpha_1 = {(1,0): 1, (1,1): 1}
    beta_1 = {(0,0): 1}
    
    # r=2: A00 * (B01 - B11)
    alpha_2 = {(0,0): 1}
    beta_2 = {(0,1): 1, (1,1): -1}
    
    # r=3: A11 * (B10 - B00)
    alpha_3 = {(1,1): 1}
    beta_3 = {(1,0): 1, (0,0): -1}
    
    # r=4: (A00 + A01) * B11
    alpha_4 = {(0,0): 1, (0,1): 1}
    beta_4 = {(1,1): 1}
    
    # r=5: (A10 - A00) * (B00 + B01)
    alpha_5 = {(1,0): 1, (0,0): -1}
    beta_5 = {(0,0): 1, (0,1): 1}
    
    # r=6: (A01 - A11) * (B10 + B11)
    alpha_6 = {(0,1): 1, (1,1): -1}
    beta_6 = {(1,0): 1, (1,1): 1}
    
    alphas = [alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6]
    betas = [beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6]
    
    # Reconstruction: C[u, w] = sum_r gamma_{r, u, w} M_r
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # gamma[r][u][w]
    gammas = [
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}, # r=0 contributes to C00, C11
        {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}, # r=1 contributes to C10, C11
        {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1},  # r=2 contributes to C01, C11
        {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0},  # r=3 contributes to C00, C10
        {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}, # r=4 contributes to C00, C01
        {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1},  # r=5 contributes to C11
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},  # r=6 contributes to C00
    ]
    # Correcting gamma based on formulas:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    gammas[0] = {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1} # M0 in C00, C11
    gammas[1] = {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1} # M1 in C10, C11
    gammas[2] = {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}  # M2 in C01, C11
    gammas[3] = {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}  # M3 in C00, C10
    gammas[4] = {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0} # M4 in C00, C01
    gammas[5] = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}  # M5 in C11
    gammas[6] = {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}  # M6 in C00
    
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for r in range(7):
                    # Column index for this specific block pair and Strassen product
                    # (I, J, K) determines the block.
                    # We need a unique column for each (I, J, K, r).
                    # Total columns = 9 * 7 = 63.
                    col = (I * num_blocks + J) * (num_blocks * 7) + K * 7 + r
                    
                    # Fill U
                    for (u, v), val in alphas[r].items():
                        i = I * block_size + u
                        j = J * block_size + v
                        U[i * m + j][col] += val
                    
                    # Fill V
                    for (v, w), val in betas[r].items():
                        j = J * block_size + v
                        k = K * block_size + w
                        V[j * p + k][col] += val
                    
                    # Fill W
                    for (u, w), val in gammas[r].items():
                        k = K * block_size + w
                        i = I * block_size + u
                        W[k * n + i][col] += val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

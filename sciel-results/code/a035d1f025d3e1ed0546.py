"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Part 1: 2x1x10 (Rank 20)
    # Part 2: 2x2x10 (Rank 35) using 5 Strassen blocks
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_1 (2x1) * B_1 (1x10)
    # A_1 uses j=0. B_1 uses j=0.
    # C_1 is full 2x10.
    col = 0
    for i in range(n):
        for k in range(p):
            # r = col
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_2 (2x2) * B_2 (2x10)
    # A_2 uses j=1,2. B_2 uses j=1,2.
    # Split B_2 columns into 5 groups of 2.
    # Strassen coefficients
    # u_s[i, j_local], v_s[j_local, k_local], w_s[i, k_local]
    # i, j_local, k_local in {0,1}
    
    # u_s: 7 x 2 x 2
    # v_s: 7 x 2 x 2
    # w_s: 7 x 2 x 2
    
    # s=1: M1
    u1 = [[1, 0], [0, 1]] # A00, A11
    v1 = [[1, 0], [0, 1]] # B00, B11
    w1 = [[1, 0], [0, 1]] # C00, C11
    
    # s=2: M2
    u2 = [[0, 0], [1, 1]] # A10, A11
    v2 = [[1, 0], [0, 0]] # B00
    w2 = [[0, 0], [1, -1]] # C10, C11
    
    # s=3: M3
    u3 = [[1, 0], [0, 0]] # A00
    v3 = [[0, 0], [1, -1]] # B10, B11
    w3 = [[0, 1], [0, 1]] # C01, C11
    
    # s=4: M4
    u4 = [[0, 0], [0, 1]] # A11
    v4 = [[-1, 1], [0, 0]] # B00, B01
    w4 = [[1, 0], [1, 0]] # C00, C10
    
    # s=5: M5
    u5 = [[1, 1], [0, 0]] # A00, A01
    v5 = [[0, 0], [0, 1]] # B11
    w5 = [[-1, 1], [0, 0]] # C00, C01
    
    # s=6: M6
    u6 = [[-1, 0], [1, 0]] # A00, A10
    v6 = [[1, 1], [0, 0]] # B00, B01
    w6 = [[0, 0], [0, 1]] # C11
    
    # s=7: M7
    u7 = [[0, 1], [-1, 0]] # A01, A11
    v7 = [[0, 0], [1, 1]] # B10, B11
    w7 = [[1, 0], [0, 0]] # C00
    
    strassen_u = [u1, u2, u3, u4, u5, u6, u7]
    strassen_v = [v1, v2, v3, v4, v5, v6, v7]
    strassen_w = [w1, w2, w3, w4, w5, w6, w7]
    
    num_groups = p // 2
    for g in range(num_groups):
        base = 20 + g * 7
        for s in range(7):
            r = base + s
            us = strassen_u[s]
            vs = strassen_v[s]
            ws = strassen_w[s]
            
            # Fill U
            for i in range(2):
                for jl in range(2):
                    j_global = 1 + jl
                    if us[i][jl] != 0:
                        U[i * m + j_global][r] += us[i][jl]
                        
            # Fill V
            for jl in range(2):
                j_global = 1 + jl
                for kl in range(2):
                    k_global = 2 * g + kl
                    if vs[jl][kl] != 0:
                        V[j_global * p + k_global][r] += vs[jl][kl]
                        
            # Fill W
            for i in range(2):
                for kl in range(2):
                    k_global = 2 * g + kl
                    if ws[i][kl] != 0:
                        W[k_global * n + i][r] += ws[i][kl]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

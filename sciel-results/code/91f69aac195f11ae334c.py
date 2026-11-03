"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Kronecker product of Strassen
    rank = 49
    
    # Strassen factors for 2x2 matrices
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices: 0=00, 1=01, 2=10, 3=11
    U_S = [
        [1, 0, 1, 0, 1, -1, 0], # 00
        [0, 0, 0, 0, 1, 0, 1],  # 01
        [0, 1, 0, 0, 0, 1, 0],  # 10
        [1, 1, 0, 1, 0, 0, -1]  # 11
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0], # 00
        [0, 0, 1, 0, 0, 1, 0],  # 01
        [0, 0, 0, 1, 0, 0, 1],  # 10
        [1, 0, -1, 0, 1, 0, 1]  # 11
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1], # 00
        [0, 0, 1, 0, 1, 0, 0],  # 01
        [0, 1, 0, 1, 0, 0, 0],  # 10
        [1, -1, 1, 0, 0, 1, 0]  # 11
    ]
    
    # Block-level Strassen coefficients
    # alpha[b][ub][vb] for X_b = sum alpha A_uv
    # beta[b][ub][vb] for Y_b = sum beta B_uv
    # gamma[b][ub][vb] for C_uv = sum gamma Z_b
    # b in 0..6, u,v in 0..1
    
    alpha = [
        [[1, 0], [0, 1]], # b=0: A00 + A11
        [[0, 0], [1, 1]], # b=1: A10 + A11
        [[1, 0], [0, 0]], # b=2: A00
        [[0, 0], [0, 1]], # b=3: A11
        [[1, 1], [0, 0]], # b=4: A00 + A01
        [[-1, 0], [1, 0]],# b=5: A10 - A00
        [[0, 1], [0, -1]] # b=6: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]], # b=0: B00 + B11
        [[1, 0], [0, 0]], # b=1: B00
        [[0, 1], [0, -1]],# b=2: B01 - B11
        [[-1, 0], [1, 0]],# b=3: B10 - B00
        [[0, 0], [0, 1]], # b=4: B11
        [[1, 1], [0, 0]], # b=5: B00 + B01
        [[0, 0], [1, 1]]  # b=6: B10 + B11
    ]
    
    gamma = [
        [[1, 0], [0, 1]], # b=0: C00+=Z0, C11+=Z0
        [[0, 0], [1, -1]],# b=1: C10+=Z1, C11-=Z1
        [[0, 1], [0, 1]], # b=2: C01+=Z2, C11+=Z2
        [[1, 0], [1, 0]], # b=3: C00+=Z3, C10+=Z3
        [[-1, 1], [0, 0]],# b=4: C00-=Z4, C01+=Z4
        [[0, 0], [0, 1]], # b=5: C11+=Z5
        [[1, 0], [0, 0]]  # b=6: C00+=Z6
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate U
    # Global A index: i * m + j. i = ib * 2 + p, j = jb * 2 + l
    for i in range(n):
        for j in range(m):
            ib = i // 2
            p = i % 2
            jb = j // 2
            l = j % 2
            idx_A = i * m + j
            idx_US = p * 2 + l
            
            for b in range(7):
                a_val = alpha[b][ib][jb]
                if a_val != 0:
                    for r in range(7):
                        col = b * 7 + r
                        u_val = U_S[idx_US][r]
                        if u_val != 0:
                            U[idx_A][col] += a_val * u_val
    
    # Populate V
    # Global B index: j * p + k. j = jb * 2 + l, k = kb * 2 + q
    for j in range(m):
        for k in range(p):
            jb = j // 2
            l = j % 2
            kb = k // 2
            q = k % 2
            idx_B = j * p + k
            idx_VS = l * 2 + q
            
            for b in range(7):
                b_val = beta[b][jb][kb]
                if b_val != 0:
                    for r in range(7):
                        col = b * 7 + r
                        v_val = V_S[idx_VS][r]
                        if v_val != 0:
                            V[idx_B][col] += b_val * v_val
    
    # Populate W
    # Global C index: k * n + i. k = kb * 2 + q, i = ib * 2 + p
    # Task convention: W[k * n + i]
    for k in range(p):
        for i in range(n):
            kb = k // 2
            q = k % 2
            ib = i // 2
            p = i % 2
            idx_C = k * n + i
            idx_WS = p * 2 + q
            
            for b in range(7):
                g_val = gamma[b][ib][kb] # Note: C_uv where u=ib, v=kb
                if g_val != 0:
                    for r in range(7):
                        col = b * 7 + r
                        w_val = W_S[idx_WS][r]
                        if w_val != 0:
                            W[idx_C][col] += g_val * w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

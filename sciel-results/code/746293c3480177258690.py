"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen coefficients for 2x2 matrix multiplication
    # u_s: coefficients for X elements [x00, x01, x10, x11]
    # v_s: coefficients for Y elements [y00, y01, y10, y11]
    inner_u = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, 0, 1, -1],# P6
        [0, 1, 0, -1] # P7
    ]
    inner_v = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 0, 1, -1],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # Inner reconstruction weights for 2x2 result R
    # R[u,v] = sum_s inner_w[s][u,v] * P_s
    # R00 = P1 + P4 - P5 + P7
    # R01 = P3 + P5
    # R10 = P2 + P4
    # R11 = P1 - P2 + P3 + P6
    inner_w = [
        # s=0 (P1)
        [[1, 0], [0, 1]],
        # s=1 (P2)
        [[0, 0], [1, -1]],
        # s=2 (P3)
        [[0, 1], [0, 1]],
        # s=3 (P4)
        [[1, 0], [1, 0]],
        # s=4 (P5)
        [[-1, 1], [0, 0]],
        # s=5 (P6)
        [[0, 0], [0, 1]],
        # s=6 (P7)
        [[1, 0], [0, 0]]
    ]
    
    # Outer Strassen block combinations
    # alpha_t: coefficients for A blocks [A00, A01, A10, A11]
    # beta_t: coefficients for B blocks [B00, B01, B10, B11]
    outer_alpha = [
        [1, 0, 0, 1], # T0: A00 + A11
        [0, 0, 1, 1], # T1: A10 + A11
        [1, 0, 0, 0], # T2: A00
        [0, 0, 0, 1], # T3: A11
        [1, 1, 0, 0], # T4: A00 + A01
        [0, 0, 1, -1],# T5: A10 - A00
        [0, 1, 0, -1] # T6: A01 - A11
    ]
    outer_beta = [
        [1, 0, 0, 1], # T0: B00 + B11
        [1, 0, 0, 0], # T1: B00
        [0, 1, 0, -1],# T2: B01 - B11
        [0, 0, 1, -1],# T3: B10 - B00
        [0, 0, 0, 1], # T4: B11
        [1, 1, 0, 0], # T5: B00 + B01
        [0, 0, 1, 1]  # T6: B10 + B11
    ]
    
    # Outer reconstruction weights for 4x4 result C
    # C_block[u,v] = sum_t outer_w[t][block, u, v] * M_t[u,v]
    # C00 = M0 - M4 + M3 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    # outer_w[t][block_idx][u,v]
    # block 0: C00, block 1: C01, block 2: C10, block 3: C11
    outer_w = [
        # t=0
        [[[1, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]]],
        # t=1
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]], [[-1, 0], [0, 0]]],
        # t=2
        [[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]]],
        # t=3
        [[[1, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]],
        # t=4
        [[[-1, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        # t=5
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]]],
        # t=6
        [[[1, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    ]
    
    # Mapping of block indices to global indices
    # Block (bi, bj) where bi, bj in {0, 1}
    # Global row i = bi * 2 + u
    # Global col j = bj * 2 + v
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # Get coefficients for inner step s
            u_coeffs = inner_u[s]
            v_coeffs = inner_v[s]
            
            # Get coefficients for outer step t
            a_coeffs = outer_alpha[t]
            b_coeffs = outer_beta[t]
            
            # Iterate over all elements of A (i, j)
            for i in range(4):
                for j in range(4):
                    # Determine which block (bi, bj) this element belongs to
                    bi = i // 2
                    bj = j // 2
                    # Local position in block
                    u_local = i % 2
                    v_local = j % 2
                    
                    # Coefficient of A[i,j] in X[u_local, v_local]
                    # X is formed by sum of blocks
                    # X[u_local, v_local] = sum_{blocks} alpha[block] * A_block[u_local, v_local]
                    # Only the block (bi, bj) contributes to A[i,j] in X[u_local, v_local]
                    # Wait, X is a 2x2 matrix.
                    # X[u_local, v_local] takes value from the same position in the contributing blocks.
                    # So if A[i,j] is in block (bi, bj), it contributes to X[u_local, v_local]
                    # with coefficient a_coeffs[bi*2 + bj].
                    
                    coeff_a = a_coeffs[bi * 2 + bj]
                    
                    # The inner step s uses u_coeffs[u_local*2 + v_local] for X[u_local, v_local]
                    # So the total coefficient for U[i*m+j, r] is:
                    u_val = coeff_a * u_coeffs[u_local * 2 + v_local]
                    U[i * m + j][r] += u_val
                    
            # Iterate over all elements of B (j, k)
            for j in range(4):
                for k in range(4):
                    # Determine which block (bj, bk) this element belongs to
                    # B is indexed by row j, col k
                    bj = j // 2
                    bk = k // 2
                    u_local = j % 2
                    v_local = k % 2
                    
                    coeff_b = b_coeffs[bj * 2 + bk]
                    
                    # The inner step s uses v_coeffs[u_local*2 + v_local] for Y[u_local, v_local]
                    v_val = coeff_b * v_coeffs[u_local * 2 + v_local]
                    V[j * p + k][r] += v_val
                    
            # Iterate over all elements of C (i, k)
            for i in range(4):
                for k in range(4):
                    # Determine which block (ci, ck) this element belongs to
                    ci = i // 2
                    ck = k // 2
                    u_local = i % 2
                    v_local = k % 2
                    
                    # The outer reconstruction for block (ci, ck) at position (u_local, v_local)
                    # is sum_t outer_w[t][block_idx][u_local, v_local] * M_t[u_local, v_local]
                    # M_t[u_local, v_local] = sum_s inner_w[s][u_local, v_local] * P_s
                    # So the coefficient for P_s (which is r) is:
                    # outer_w[t][block_idx][u_local, v_local] * inner_w[s][u_local, v_local]
                    
                    block_idx = ci * 2 + ck
                    w_val = outer_w[t][block_idx][u_local][v_local] * inner_w[s][u_local][v_local]
                    W[k * n + i][r] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen on 2x2 blocks
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen coefficients for 2x2 multiplication
    # P_l = (sum u_ab x_ab) (sum v_cd y_cd)
    # u, v are 2x2 matrices of coefficients
    inner_u = [
        [[1, 0], [0, 1]],  # P1: x00 + x11
        [[0, 0], [1, 1]],  # P2: x10 + x11
        [[1, 0], [0, 0]],  # P3: x00
        [[0, 0], [0, 1]],  # P4: x11
        [[1, 1], [0, 0]],  # P5: x00 + x01
        [[-1, 0], [1, 0]], # P6: x10 - x00
        [[0, 1], [0, -1]]  # P7: x01 - x11
    ]
    inner_v = [
        [[1, 0], [0, 1]],  # P1: y00 + y11
        [[1, 0], [0, 0]],  # P2: y00
        [[0, 1], [0, -1]], # P3: y01 - y11
        [[0, -1], [1, 0]], # P4: y10 - y00
        [[0, 0], [0, 1]],  # P5: y11
        [[1, 1], [0, 0]],  # P6: y00 + y01
        [[0, 0], [1, 1]]   # P7: y10 + y11
    ]
    
    # Output combination for Inner Strassen
    # Z_ab = sum_l eta_l_ab P_l
    # eta is 2x2 for each l
    inner_eta = [
        [[1, 0], [0, 1]],  # P1 contributes to Z00, Z11
        [[0, 0], [1, -1]], # P2 contributes to Z10, Z11
        [[0, 1], [0, 1]],  # P3 contributes to Z01, Z11
        [[1, 0], [1, 0]],  # P4 contributes to Z00, Z10
        [[-1, 0], [0, 1]], # P5 contributes to Z00, Z01
        [[0, 0], [0, 1]],  # P6 contributes to Z11
        [[1, 0], [0, 0]]   # P7 contributes to Z00
    ]
    # Wait, let's verify Inner Strassen output formulas:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # Let's re-define inner_eta correctly
    # eta[l][r][c] is coeff of P_l in Z[r][c]
    inner_eta = [
        [[1, 0], [0, 1]],  # P1: Z00+=1, Z11+=1
        [[0, 0], [1, -1]], # P2: Z10+=1, Z11+=-1
        [[0, 1], [0, 1]],  # P3: Z01+=1, Z11+=1
        [[1, 0], [1, 0]],  # P4: Z00+=1, Z10+=1
        [[-1, 1], [0, 0]], # P5: Z00+=-1, Z01+=1
        [[0, 0], [0, 1]],  # P6: Z11+=1
        [[1, 0], [0, 0]]   # P7: Z00+=1
    ]
    
    # Outer Strassen coefficients for 4x4 multiplication (on 2x2 blocks)
    # M_k = X_k Y_k
    # X_k = sum_blocks alpha_k_block A_block
    # Y_k = sum_blocks beta_k_block B_block
    
    # Blocks: 00, 01, 10, 11
    # alpha[k][block]
    outer_alpha = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    outer_beta = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    # Output combination for Outer Strassen
    # C_block_rc = sum_k delta_k_block_rc M_k
    # delta[k][block_r][block_c]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    outer_delta = [
        [[1, 0], [0, 1]],  # M1: C00+=1, C11+=1
        [[0, 0], [1, -1]], # M2: C10+=1, C11+=-1
        [[0, 1], [0, 1]],  # M3: C01+=1, C11+=1
        [[1, 0], [1, 0]],  # M4: C00+=1, C10+=1
        [[-1, 1], [0, 0]], # M5: C00+=-1, C01+=1
        [[0, 0], [0, 1]],  # M6: C11+=1
        [[1, 0], [0, 0]]   # M7: C00+=1
    ]
    
    # Helper to get flat index for A[i][j]
    def a_idx(i, j):
        return i * m + j
        
    # Helper to get flat index for B[i][j]
    def b_idx(i, j):
        return i * p + j
        
    # Helper to get flat index for C[i][j]
    def c_idx(i, j):
        return j * n + i # W[k*n+i, r] convention: index is k*n+i. Here i is row, k is col.
        # Wait, prompt says W[k*n+i, r].
        # In matrix mult C[i][k] = sum_j A[i][j]B[j][k].
        # So C index is (i, k).
        # Prompt says W[k*n+i, r].
        # This implies the first index of W is k*n+i.
        # Let's check schoolbook: W[k*n+i][column] = 1.
        # So yes, W index is k*n+i.
        
    col = 0
    for k in range(7): # Outer product index
        for l in range(7): # Inner product index
            # Construct U and V coefficients for this column
            # U corresponds to A
            # V corresponds to B
            
            # U_flat[i*m+j] = coeff of A[i][j] in L_X
            # L_X = sum_ab inner_u[l][a][b] * X_k[a][b]
            # X_k[a][b] = sum_block outer_alpha[k][block] * A_block[a][b]
            
            u_coeffs = [0] * (n * m)
            v_coeffs = [0] * (m * p)
            
            for a in range(2):
                for b in range(2):
                    coeff_inner = inner_u[l][a][b]
                    if coeff_inner == 0:
                        continue
                    for block_idx in range(4):
                        coeff_outer = outer_alpha[k][block_idx]
                        if coeff_outer == 0:
                            continue
                        coeff_total = coeff_inner * coeff_outer
                        if coeff_total == 0:
                            continue
                        
                        # Determine A indices
                        # block_idx 00: rows 0-1, cols 0-1
                        # block_idx 01: rows 0-1, cols 2-3
                        # block_idx 10: rows 2-3, cols 0-1
                        # block_idx 11: rows 2-3, cols 2-3
                        
                        block_r = block_idx // 2
                        block_c = block_idx % 2
                        
                        row_offset = block_r * 2
                        col_offset = block_c * 2
                        
                        i = row_offset + a
                        j = col_offset + b
                        
                        u_coeffs[a_idx(i, j)] += coeff_total
            
            # Similarly for V
            for c in range(2):
                for d in range(2):
                    coeff_inner = inner_v[l][c][d]
                    if coeff_inner == 0:
                        continue
                    for block_idx in range(4):
                        coeff_outer = outer_beta[k][block_idx]
                        if coeff_outer == 0:
                            continue
                        coeff_total = coeff_inner * coeff_outer
                        if coeff_total == 0:
                            continue
                        
                        block_r = block_idx // 2
                        block_c = block_idx % 2
                        
                        row_offset = block_r * 2
                        col_offset = block_c * 2
                        
                        i = row_offset + c
                        j = col_offset + d
                        
                        v_coeffs[b_idx(i, j)] += coeff_total
            
            # Set U and V for this column
            for idx in range(n * m):
                U[idx][col] = u_coeffs[idx]
            for idx in range(m * p):
                V[idx][col] = v_coeffs[idx]
                
            # Construct W coefficients
            # W[k*n+i, r] = coeff of P_kl in C[i][k]
            # C[i][k] = sum_K outer_delta[K][block_i][block_k] * (M_K)[row_i][col_k]
            # (M_K)[row_i][col_k] = sum_L inner_eta[L][row_i][col_k] * P_KL
            
            # We are at column col corresponding to (K=k, L=l)
            # So we need to add to W the term:
            # outer_delta[k][block_i][block_k] * inner_eta[l][row_i][col_k]
            
            for i in range(n):
                for k_c in range(p): # k_c is column index of C
                    block_i = i // 2
                    row_i = i % 2
                    block_k = k_c // 2
                    col_k = k_c % 5 # Wait, p=4, so block_k is 0 or 1.
                    # p=4, so k_c in 0..3.
                    # block_k = k_c // 2.
                    
                    coeff_outer = outer_delta[k][block_i][block_k]
                    if coeff_outer == 0:
                        continue
                    
                    coeff_inner = inner_eta[l][row_i][col_k]
                    if coeff_inner == 0:
                        continue
                    
                    coeff_total = coeff_outer * coeff_inner
                    
                    w_idx = k_c * n + i
                    W[w_idx][col] = coeff_total
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

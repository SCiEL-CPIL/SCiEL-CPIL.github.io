"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen inner terms (X, Y parts)
    # Each term is a tuple (X_comb, Y_comb)
    # X_comb: list of (row, col, coeff)
    # Y_comb: list of (row, col, coeff)
    
    inner_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # S1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),           # S2
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),          # S3
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),          # S4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),           # S5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # S6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])  # S7
    ]
    
    # Define Strassen outer terms (A blocks, B blocks)
    # Each term is a tuple (A_comb, B_comb)
    # A_comb: list of (block_row, block_col, coeff)
    # B_comb: list of (block_row, block_col, coeff)
    
    outer_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),           # M2
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),          # M3
        ([(1,1,1)], [(0,1,1), (0,0,-1)]),          # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),           # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])  # M7
    ]
    
    # Reconstruction matrices
    # W_inner[r, c, s] is coeff of S_s in Z_r,c
    W_inner = [
        [ [1, 0, 0, 1, -1, 0, 1],  [0, 0, 1, 0, 1, 0, 0] ], # Z00, Z01
        [ [0, 1, 0, 1, 0, 0, 0],   [1, -1, 1, 0, 0, 1, 0] ]  # Z10, Z11
    ]
    # Wait, Z11 = S1 - S2 + S3 + S6.
    # My W_inner[1][1] is for Z11.
    # Let's check indices:
    # Z00: S1+S4-S5+S7 -> [1, 0, 0, 1, -1, 0, 1]
    # Z01: S3+S5       -> [0, 0, 1, 0, 1, 0, 0]
    # Z10: S2+S4       -> [0, 1, 0, 1, 0, 0, 0]
    # Z11: S1-S2+S3+S6 -> [1, -1, 1, 0, 0, 1, 0]
    
    # W_outer[r, c, m] is coeff of M_m in C_block_r,c
    # C00 (block 0,0): M1+M4-M5+M7
    # C01 (block 0,1): M3+M5
    # C10 (block 1,0): M2+M4
    # C11 (block 1,1): M1-M2+M3+M6
    W_outer = [
        [ [1, 0, 0, 1, -1, 0, 1],  [0, 0, 1, 0, 1, 0, 0] ], # Block 00, 01
        [ [0, 1, 0, 1, 0, 0, 0],   [1, -1, 1, 0, 0, 1, 0] ]  # Block 10, 11
    ]
    # Note: W_outer and W_inner have the same structure because Strassen is the same.
    
    col = 0
    for m_out in range(7):
        A_comb_outer, B_comb_outer = outer_terms[m_out]
        for m_in in range(7):
            X_comb_inner, Y_comb_inner = inner_terms[m_in]
            
            # Build U and V contributions
            # U index: i * m + j
            # V index: j * p + k
            
            # We need to sum contributions to U and V for this rank-1 term.
            # U_val[i, j] += coeff_A
            # V_val[j, k] += coeff_B
            # W_val[k, i] += coeff_W
            
            U_val = [[0]*m for _ in range(n)]
            V_val = [[0]*p for _ in range(m)]
            W_val = [[0]*n for _ in range(p)]
            
            # Determine A coefficients
            # A_comb_outer defines which blocks are used.
            # X_comb_inner defines which elements within those blocks are used.
            # The total coefficient for A[i, j] is sum over matching block and local indices.
            
            for (br, bc, c_a) in A_comb_outer:
                for (lr, lc, c_x) in X_comb_inner:
                    i = 2 * br + lr
                    j = 2 * bc + lc
                    U_val[i][j] += c_a * c_x
            
            # Determine B coefficients
            for (br, bc, c_b) in B_comb_outer:
                for (lr, lc, c_y) in Y_comb_inner:
                    j = 2 * br + lr
                    k = 2 * bc + lc
                    V_val[j][k] += c_b * c_y
            
            # Determine W coefficients
            # W_val[k, i] accumulates the coefficient for C[k, i].
            # C_block[br, bc] is composed of M_out[m_out] with coeff W_outer[br, bc, m_out].
            # M_out[m_out][lr, lc] is composed of S_in[m_in] with coeff W_inner[lr, lc, m_in].
            # So C[2*br+lc, 2*bc+lr] += W_outer[br, bc, m_out] * W_inner[lr, lc, m_in]
            # Wait, C index is (row, col).
            # Block row br, block col bc.
            # Local row lr, local col lc.
            # Global row i = 2*br + lr.
            # Global col k = 2*bc + lc.
            # W matrix in the problem statement is W[k*n+i, r].
            # So W_val should map to (k, i).
            
            for br in range(2):
                for bc in range(2):
                    c_outer = W_outer[br][bc][m_out]
                    if c_outer == 0: continue
                    for lr in range(2):
                        for lc in range(2):
                            c_inner = W_inner[lr][lc][m_in]
                            if c_inner == 0: continue
                            i = 2 * br + lr
                            k = 2 * bc + lc
                            W_val[k][i] += c_outer * c_inner
            
            # Write to U, V, W
            for i in range(n):
                for j in range(m):
                    U[i*m+j][col] = U_val[i][j]
            for j in range(m):
                for k in range(p):
                    V[j*p+k][col] = V_val[j][k]
            for k in range(p):
                for i in range(n):
                    W[k*n+i][col] = W_val[k][i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

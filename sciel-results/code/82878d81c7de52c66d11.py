"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition of rank 42 for 2x2x12 matrix multiplication.
    # The problem decomposes into 6 independent 2x2x2 multiplications.
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 6 blocks
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Indices in A (2x2)
    # A = [[a, b], [c, d]]
    # Index: row * m + col
    idx_a = 0 * m + 0
    idx_b = 0 * m + 1
    idx_c = 1 * m + 0
    idx_d = 1 * m + 1
    
    for k in range(num_blocks):
        # Base column index for this block
        base_col = k * rank_per_block
        
        # Indices in B (2x12) for block k
        # B rows 0 and 1, cols 2k and 2k+1
        # Index: row * p + col
        col_b0 = 2 * k
        col_b1 = 2 * k + 1
        idx_e = 0 * p + col_b0  # B[0, 2k]
        idx_f = 0 * p + col_b1  # B[0, 2k+1]
        idx_g = 1 * p + col_b0  # B[1, 2k]
        idx_h = 1 * p + col_b1  # B[1, 2k+1]
        
        # Indices in C (2x12) for block k
        # C rows 0 and 1, cols 2k and 2k+1
        # Index: col * n + row
        idx_i = col_b0 * n + 0  # C[0, 2k]
        idx_k_out = col_b0 * n + 1  # C[1, 2k]
        idx_j = col_b1 * n + 0  # C[0, 2k+1]
        idx_l = col_b1 * n + 1  # C[1, 2k+1]
        
        # Strassen products M1..M7
        # M1 = (a+d)(e+h)
        r = base_col + 0
        U[idx_a][r] = 1
        U[idx_d][r] = 1
        V[idx_e][r] = 1
        V[idx_h][r] = 1
        W[idx_i][r] = 1
        W[idx_l][r] = 1
        
        # M2 = (c+d)e
        r = base_col + 1
        U[idx_c][r] = 1
        U[idx_d][r] = 1
        V[idx_e][r] = 1
        W[idx_k_out][r] = 1
        W[idx_l][r] = -1
        
        # M3 = a(f-h)
        r = base_col + 2
        U[idx_a][r] = 1
        V[idx_f][r] = 1
        V[idx_h][r] = -1
        W[idx_j][r] = 1
        W[idx_l][r] = 1
        
        # M4 = d(g-e)
        r = base_col + 3
        U[idx_d][r] = 1
        V[idx_g][r] = 1
        V[idx_e][r] = -1
        W[idx_i][r] = 1
        W[idx_k_out][r] = 1
        
        # M5 = (a+b)h
        r = base_col + 4
        U[idx_a][r] = 1
        U[idx_b][r] = 1
        V[idx_h][r] = 1
        W[idx_i][r] = -1
        W[idx_j][r] = 1
        
        # M6 = (c-a)(e+f)
        r = base_col + 5
        U[idx_c][r] = 1
        U[idx_a][r] = -1
        V[idx_e][r] = 1
        V[idx_f][r] = 1
        W[idx_l][r] = 1
        
        # M7 = (b-d)(g+h)
        r = base_col + 6
        U[idx_b][r] = 1
        U[idx_d][r] = -1
        V[idx_g][r] = 1
        V[idx_h][r] = 1
        W[idx_i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

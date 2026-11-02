"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block decomposition: treat 6x6 as 2x2 blocks of 3x3 matrices
    # Use Strassen's algorithm for the 2x2 block structure (rank 7)
    # Each term involves 3x3 matrix multiplication (rank 27, schoolbook)
    # Total rank: 7 * 27 = 189
    
    block_size = 3
    rank_per_block_mult = block_size ** 3  # 27
    strassen_rank = 7
    total_rank = strassen_rank * rank_per_block_mult  # 189
    
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen 2x2 decomposition:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A * B where each entry is a 3x3 matrix
    #
    # Strassen products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the linear combinations for each Strassen term
    # Each term: (A_comb, B_comb) where A_comb and B_comb are lists of (block_idx, coeff)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 for A; same for B
    
    strassen_terms = [
        # (A_comb, B_comb, W_comb)
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (1, -1), (2, 1)]),  # C00, C11
        # M2 = (A10 + A11) * B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),  # C10, C11
        # M3 = A00 * (B01 - B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),  # C01, C11
        # M4 = A11 * (B10 - B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),  # C00, C10
        # M5 = (A00 + A01) * B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),  # C00, C01
        # M6 = (A10 - A00) * (B00 + B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),  # C11
        # M7 = (A01 - A11) * (B10 + B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),  # C00
    ]
    
    # For each Strassen term, we need to do 3x3 matrix multiplication
    # The 3x3 multiplication is schoolbook: rank 27
    # For each of the 27 scalar multiplications in the 3x3 mult,
    # we add the appropriate coefficients to U, V, W
    
    col = 0
    for term_idx in range(7):
        A_comb, B_comb, W_comb = strassen_terms[term_idx]
        
        # For each of the 27 elements in the 3x3 multiplication
        for i in range(block_size):
            for j in range(block_size):
                for k in range(block_size):
                    # This corresponds to the (i,j) element of the 3x3 product
                    # which comes from summing over k: A[i,k] * B[k,j]
                    
                    # For the A factor:
                    # The A block is a linear combination of the 4 A blocks
                    # A[i,k] in the combined A matrix comes from:
                    # For each (block_idx, coeff) in A_comb:
                    #   A_block[block_idx][i,k] * coeff
                    # In the full 6x6 matrix A, block (bi, bj) is at rows bi*3 to bi*3+2, cols bj*3 to bj*3+2
                    # Block 0 = A00: rows 0-2, cols 0-2
                    # Block 1 = A01: rows 0-2, cols 3-5
                    # Block 2 = A10: rows 3-5, cols 0-2
                    # Block 3 = A11: rows 3-5, cols 3-5
                    
                    # U index: i_full * m + j_full where i_full is row, j_full is col
                    # For A: i_full = bi * 3 + i, j_full = bj * 3 + k
                    # U[i_full * m + j_full]
                    
                    # V index: j_full * p + k_full where j_full is row, k_full is col
                    # For B: j_full = bj * 3 + k, k_full = bk * 3 + j
                    # V[j_full * p + k_full]
                    
                    # W index: k_full * n + i_full where k_full is row, i_full is col
                    # For C: k_full = bk * 3 + j, i_full = bi * 3 + i
                    # W[k_full * n + i_full]
                    
                    # Block mapping:
                    # Block 0: bi=0, bj=0
                    # Block 1: bi=0, bj=1
                    # Block 2: bi=1, bj=0
                    # Block 3: bi=1, bj=1
                    
                    block_to_bi_bj = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
                    
                    # Add contributions from A_comb
                    for (a_block, a_coeff) in A_comb:
                        bi, bj = block_to_bi_bj[a_block]
                        u_idx = (bi * block_size + i) * m + (bj * block_size + k)
                        U[u_idx][col] += a_coeff
                    
                    # Add contributions from B_comb
                    for (b_block, b_coeff) in B_comb:
                        bi, bj = block_to_bi_bj[b_block]
                        v_idx = (bi * block_size + k) * p + (bj * block_size + j)
                        V[v_idx][col] += b_coeff
                    
                    # Add contributions from W_comb
                    for (c_block, c_coeff) in W_comb:
                        bi, bj = block_to_bi_bj[c_block]
                        w_idx = (bi * block_size + j) * n + (bj * block_size + i)
                        W[w_idx][col] += c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

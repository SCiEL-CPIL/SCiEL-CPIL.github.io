import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 2, number of blocks 3
    block_size = 2
    num_blocks = n // block_size
    
    rank = (num_blocks ** 3) * 7  # 27 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U coeffs: [a, b, c, d]
    # V coeffs: [e, f, g, h]
    # W coeffs: [C00, C01, C10, C11]
    strassen_coeffs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # P3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # P7
    ]
    
    # Indices for local 2x2 blocks
    # A[I, J] local indices
    # A[0,0] -> (0,0), A[0,1] -> (0,1), A[1,0] -> (1,0), A[1,1] -> (1,1)
    # Global indices:
    # A row: I*2 + i_loc, A col: J*2 + j_loc
    # B row: J*2 + j_loc, B col: K*2 + k_loc
    # C row: I*2 + i_loc, C col: K*2 + k_loc
    
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                t = I * num_blocks * num_blocks + J * num_blocks + K
                for k in range(7):
                    r = t * 7 + k
                    u_coeffs, v_coeffs, w_coeffs = strassen_coeffs[k]
                    
                    # Update U
                    # u_coeffs corresponds to [a, b, c, d]
                    # a: i_loc=0, j_loc=0
                    # b: i_loc=0, j_loc=1
                    # c: i_loc=1, j_loc=0
                    # d: i_loc=1, j_loc=1
                    for loc_idx, coeff in enumerate(u_coeffs):
                        if coeff != 0:
                            i_loc = loc_idx // 2
                            j_loc = loc_idx % 2
                            row_A = I * block_size + i_loc
                            col_A = J * block_size + j_loc
                            U[row_A * m + col_A][r] += coeff
                            
                    # Update V
                    # v_coeffs corresponds to [e, f, g, h]
                    # e: i_loc=0, j_loc=0 (row J, col K)
                    # f: i_loc=0, j_loc=1
                    # g: i_loc=1, j_loc=0
                    # h: i_loc=1, j_loc=1
                    for loc_idx, coeff in enumerate(v_coeffs):
                        if coeff != 0:
                            i_loc = loc_idx // 2
                            j_loc = loc_idx % 2
                            row_B = J * block_size + i_loc
                            col_B = K * block_size + j_loc
                            V[row_B * p + col_B][r] += coeff
                            
                    # Update W
                    # w_coeffs corresponds to [C00, C01, C10, C11]
                    # C00: i_loc=0, k_loc=0
                    # C01: i_loc=0, k_loc=1
                    # C10: i_loc=1, k_loc=0
                    # C11: i_loc=1, k_loc=1
                    for loc_idx, coeff in enumerate(w_coeffs):
                        if coeff != 0:
                            i_loc = loc_idx // 2
                            k_loc = loc_idx % 2
                            row_C = I * block_size + i_loc
                            col_C = K * block_size + k_loc
                            # W index is k*n + i -> col_C * n + row_C
                            W[col_C * n + row_C][r] += coeff
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

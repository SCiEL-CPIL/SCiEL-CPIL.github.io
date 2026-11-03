"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms: each is (A_blocks, B_blocks, W_targets)
    # A_blocks: list of (block, coeff) for the 1x2 vector
    # B_blocks: list of (block, coeff) for the 2x3 vector
    # W_targets: dict of {c_block: sign} for the 1x3 result
    
    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
        ([(0,1),(3,1)], [(0,1),(3,1)], {0:1, 3:1}),
        # M2 = (A10+A11)*B00 -> C10(+1)
        ([(2,1),(3,1)], [(0,1)], {2:1}),
        # M3 = A00*(B01-B11) -> C01(+1), C11(-1)
        ([(0,1)], [(1,1),(3,-1)], {1:1, 3:-1}),
        # M4 = A11*(B10-B00) -> C00(+1), C10(+1)
        ([(3,1)], [(2,1),(0,-1)], {0:1, 2:1}),
        # M5 = (A00+A10)*B11 -> C01(+1), C11(+1)
        ([(0,1),(2,1)], [(3,1)], {1:1, 3:1}),
        # M6 = (A11-A10)*(B00+B10) -> C00(-1)
        ([(3,1),(2,-1)], [(0,1),(2,1)], {0:-1}),
        # M7 = (A01-A11)*(B01+B11) -> C00(+1), C11(+1)
        ([(1,1),(3,-1)], [(1,1),(3,1)], {0:1, 3:1}),
    ]
    
    col = 0
    for A_blocks, B_blocks, W_targets in strassen_terms:
        for j_sub in range(2):
            for k_sub in range(3):
                # U: for each A block, set U at the j_sub-th column
                for (a_block, ca) in A_blocks:
                    u_idx = 2 * a_block + j_sub
                    U[u_idx][col] += ca
                
                # V: for each B block, set V at (j_sub, k_sub)
                for (b_block, cb) in B_blocks:
                    v_idx = ((b_block // 2) * 2 + j_sub) * 6 + (b_block % 2) * 3 + k_sub
                    V[v_idx][col] += cb
                
                # W: for each target C block, set W at k_sub-th column
                for c_block, sign in W_targets.items():
                    c_row = c_block // 2
                    c_col_block = c_block % 2
                    w_idx = (c_col_block * 3 + k_sub) * 2 + c_row
                    W[w_idx][col] += sign
                
                col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

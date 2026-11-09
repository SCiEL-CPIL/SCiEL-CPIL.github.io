"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen at block level (2x2 blocks of 3x3), schoolbook at 3x3 level
    # Total rank = 7 * 27 = 189
    
    # Define the 7 Strassen products
    # Each product: (A_support, B_support, output_contributions)
    # A_support: list of (block_row, block_col, coeff)
    # B_support: list of (block_row, block_col, coeff)
    # output_contributions: list of (block_row, block_col, coeff)
    
    strassen_products = [
        # M1: (A00+A11)(B00+B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)B01
        ([(1,0,1), (1,1,1)], [(0,1,1)], [(0,1,1)]),
        # M3: A00(B10-B11)
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(1,0,1)]),
        # M4: A11(B00-B10)
        ([(1,1,1)], [(0,0,1), (1,0,-1)], [(0,0,1), (1,1,1)]),
        # M5: (A00+A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6: (A10-A00)(B00+B10)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,0,1)]),
        # M7: (A01-A11)(B10+B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    block_size = 3
    num_blocks = 2
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod_idx in range(7):
        A_support, B_support, out_contribs = strassen_products[prod_idx]
        for di in range(block_size):
            for dj in range(block_size):
                for dk in range(block_size):
                    # Set U entries
                    for (ai, aj, coeff) in A_support:
                        global_i = ai * block_size + di
                        global_j = aj * block_size + dj
                        U[global_i * m + global_j][col] = coeff
                    # Set V entries
                    for (bi, bk, coeff) in B_support:
                        global_j = bi * block_size + dj
                        global_k = bk * block_size + dk
                        V[global_j * p + global_k][col] = coeff
                    # Set W entries
                    for (oi, ok, coeff) in out_contribs:
                        global_i = oi * block_size + di
                        global_k = ok * block_size + dk
                        W[global_k * n + global_i][col] = coeff
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

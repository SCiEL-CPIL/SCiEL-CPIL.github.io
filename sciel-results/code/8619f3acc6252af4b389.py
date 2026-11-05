"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 6x6 matrix multiplication.
    
    Partition 6x6 into 2x2 grid of 3x3 blocks.
    Use Strassen at block level (7 products), each product is 3x3 matrix mult (rank 9).
    Total rank = 7 * 9 = 63.
    
    For each Strassen product M_i = X_i * Y_i where X_i and Y_i are 3x3 matrices
    (linear combinations of A and B blocks), we use the standard rank-9 schoolbook
    decomposition: for each (di,dj,dk) in 3x3x3, we have one scalar multiplication.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size
    
    rank = 7 * bs * bs  # 63
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    # For each of the 7 Strassen products, define:
    # - A_block contributions: list of (ai, aj, coeff) for the 3x3 block
    # - B_block contributions: list of (bi, bj, coeff) for the 3x3 block  
    # - C_block contributions: list of (ci, cj, coeff) for the output 3x3 block
    # Each block is identified by (row_block_idx, col_block_idx)
    
    strassen_products = [
        # M1 = (A11+A22)(B11+B22), contributes to C11 with +1
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(0,0,1)]
        },
        # M2 = (A21+A22)B11, contributes to C21 with +1
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1)]
        },
        # M3 = A11(B12-B22), contributes to C12 with +1
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'C': [(0,1,1)]
        },
        # M4 = A22(B21-B11), contributes to C21 with +1 and C11 with +1
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'C': [(1,0,1), (0,0,1)]
        },
        # M5 = (A11+A12)B22, contributes to C11 with -1 and C12 with +1
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,0,-1), (0,1,1)]
        },
        # M6 = (A21-A11)(B11+B12), contributes to C22 with +1
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(1,1,1)]
        },
        # M7 = (A12-A22)(B21+B22), contributes to C11 with +1
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'C': [(0,0,1)]
        },
    ]
    
    for prod_idx, prod in enumerate(strassen_products):
        base_col = prod_idx * bs * bs  # 9 columns per product
        
        for di in range(bs):
            for dj in range(bs):
                for dk in range(bs):
                    col = base_col + di * bs * bs + dj * bs + dk
                    
                    # U: sum of contributions from A blocks
                    u_val = 0
                    for (ai, aj, coeff) in prod['A']:
                        u_row = (ai * bs + di) * m + (aj * bs + dj)
                        U[u_row][col] += coeff
                    
                    # V: sum of contributions from B blocks
                    v_val = 0
                    for (bi, bj, coeff) in prod['B']:
                        v_row = (bi * bs + dj) * p + (bj * bs + dk)
                        V[v_row][col] += coeff
                    
                    # W: sum of contributions from C blocks
                    for (ci, cj, coeff) in prod['C']:
                        w_row = (cj * bs + dk) * n + (ci * bs + di)
                        W[w_row][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

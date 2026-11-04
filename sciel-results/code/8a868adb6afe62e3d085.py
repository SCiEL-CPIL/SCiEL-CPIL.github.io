"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 12 = 84
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen configurations for 4x4x6
    # Each config is a tuple: (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    # Block indices:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    configs = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
    ]
    
    col = 0
    for a_blocks, b_blocks, c_blocks in configs:
        # Iterate over 12 local terms for 2x2x3
        for u in range(2):
            for v in range(2):
                for w in range(3):
                    # Local values are all 1
                    # For each A block contributing to this M
                    for a_idx, a_coeff in a_blocks:
                        a_row_block = a_idx // 2
                        a_col_block = a_idx % 2
                        # Global i, j
                        gi = 2 * a_row_block + u
                        gj = 2 * a_col_block + v
                        U[gi * m + gj][col] += a_coeff
                    
                    # For each B block contributing to this M
                    for b_idx, b_coeff in b_blocks:
                        b_row_block = b_idx // 2
                        b_col_block = b_idx % 2
                        # Global j, k
                        gj = 2 * b_row_block + v
                        gk = 3 * b_col_block + w
                        V[gj * p + gk][col] += b_coeff
                    
                    # For each C block contributing to this M
                    for c_idx, c_coeff in c_blocks:
                        c_row_block = c_idx // 2
                        c_col_block = c_idx % 2
                        # Global k, i
                        gk = 3 * c_col_block + w
                        gi = 2 * c_row_block + u
                        W[gk * n + gi][col] += c_coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

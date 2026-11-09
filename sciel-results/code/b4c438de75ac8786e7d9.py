"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: 3x3 blocks of 2x2
    # Use Strassen (rank 7) for 2x2, schoolbook (rank 27) for 3x3 block level
    # Total rank = 27 * 7 = 189
    
    # Strassen 2x2 decomposition
    # u_s, v_s, w_s are 2x2 matrices for s=1..7
    u_strassen = [
        [[1, 0], [0, 1]],   # p1 = (a+d)(e+h)
        [[0, 0], [1, 1]],   # p2 = (c+d)e
        [[1, 0], [0, 0]],   # p3 = a(f-h)
        [[0, 0], [0, 1]],   # p4 = d(g-e)
        [[1, 1], [0, 0]],   # p5 = (a+b)h
        [[-1, 0], [1, 0]],  # p6 = (c-a)(e+f)
        [[0, 1], [0, -1]],  # p7 = (b-d)(g+h)
    ]
    v_strassen = [
        [[1, 0], [0, 1]],   # (e+h)
        [[1, 0], [0, 0]],   # e
        [[0, 1], [0, -1]],  # (f-h)
        [[0, 0], [1, -1]],  # (g-e)
        [[0, 0], [0, 1]],   # h
        [[1, 1], [0, 0]],   # (e+f)
        [[0, 0], [1, 1]],   # (g+h)
    ]
    w_strassen = [
        [[1, 0], [0, 1]],   # C00 += 1, C11 += 1
        [[0, 0], [1, -1]],  # C10 += 1, C11 -= 1
        [[0, 1], [0, 1]],   # C01 += 1, C11 += 1
        [[1, 0], [1, 0]],   # C00 += 1, C10 += 1
        [[-1, 1], [0, 0]],  # C00 -= 1, C01 += 1
        [[0, 0], [0, 1]],   # C11 += 1
        [[1, 0], [0, 0]],   # C00 += 1
    ]
    
    block_size = 2
    num_blocks = 3
    rank = num_blocks * num_blocks * num_blocks * 7  # 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(num_blocks):
        for j_block in range(num_blocks):
            for k_block in range(num_blocks):
                for s in range(7):
                    # U: block (i_block, j_block), within-block from u_strassen[s]
                    for i_local in range(block_size):
                        for j_local in range(block_size):
                            row = (i_block * block_size + i_local) * m + (j_block * block_size + j_local)
                            U[row][col] = u_strassen[s][i_local][j_local]
                    
                    # V: block (j_block, k_block), within-block from v_strassen[s]
                    for j_local in range(block_size):
                        for k_local in range(block_size):
                            row = (j_block * block_size + j_local) * p + (k_block * block_size + k_local)
                            V[row][col] = v_strassen[s][j_local][k_local]
                    
                    # W: block (k_block, i_block), within-block from w_strassen[s]
                    for k_local in range(block_size):
                        for i_local in range(block_size):
                            row = (k_block * block_size + k_local) * n + (i_block * block_size + i_local)
                            W[row][col] = w_strassen[s][k_local][i_local]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

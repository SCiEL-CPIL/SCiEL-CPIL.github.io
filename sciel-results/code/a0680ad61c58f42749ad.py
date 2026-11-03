# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Inner Strassen for 2x2x2
    # Flattening order: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
    ]
    
    # Block indices for 4x4 matrix (2x2 blocks of 2x2)
    # Block (i, j) of 4x4 matrix
    # Indices in flattened 16-vector:
    # Block (0,0): 0, 1, 4, 5
    # Block (0,1): 2, 3, 6, 7
    # Block (1,0): 8, 9, 12, 13
    # Block (1,1): 10, 11, 14, 15
    
    def get_block_indices(block_i, block_j):
        start_r = block_i * 2
        start_c = block_j * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = start_r + dr
                c = start_c + dc
                indices.append(r * 4 + c)
        return indices

    block_indices = {}
    for bi in range(2):
        for bj in range(2):
            block_indices[(bi, bj)] = get_block_indices(bi, bj)

    # Outer Strassen for 4x4x4 (using 2x2 blocks)
    # Each term is defined by:
    # L_blocks: dict (bi, bj) -> coeff
    # R_blocks: dict (bi, bj) -> coeff
    # C_blocks: dict (bi, bj) -> coeff
    
    outer_strassen = [
        # M1 = (A00 + A11)(B00 + B11) -> C00 + C11
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2 = (A10 + A11) B00 -> C10 + C11
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): 1}
        ),
        # M3 = A00 (B01 - B11) -> C01 + C11
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # M4 = A11 (B10 - B00) -> C00 + C10
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # M5 = (A00 + A01) B11 -> C00 - C11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1, (1,1): -1}
        ),
        # M6 = (A10 - A00)(B00 + B01) -> C11
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7 = (A01 - A11)(B10 + B11) -> C00 + C11
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        )
    ]
    
    # Total rank: 2 (for p=8) * 7 (outer) * 7 (inner) = 98
    rank = 98
    
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 98
    
    for block in range(2):
        base_col = block * 49
        for prod_idx, (L_blocks, R_blocks, C_blocks) in enumerate(outer_strassen):
            for s_idx, (u_2, v_2, w_2) in enumerate(strassen_2x2):
                col = base_col + prod_idx * 7 + s_idx
                
                # Map u_2 to U
                # u_2 is a vector of 4 elements for a 2x2 block
                # L_blocks maps block (bi, bj) to coeff
                for (bi, bj), coeff_L in L_blocks.items():
                    u_indices = block_indices[(bi, bj)]
                    for k, idx in enumerate(u_indices):
                        U[idx][col] += u_2[k] * coeff_L
                        
                # Map v_2 to V
                # V index: j * p + k
                for (bi, bj), coeff_R in R_blocks.items():
                    v_indices = block_indices[(bi, bj)]
                    for k, idx in enumerate(v_indices):
                        b_row = idx // 4
                        b_col_local = idx % 4
                        b_col_global = block * 4 + b_col_local
                        v_flat_idx = b_row * p + b_col_global
                        V[v_flat_idx][col] += v_2[k] * coeff_R
                        
                # Map w_2 to W
                # W index: k * n + i
                for (ci, cj), coeff_C in C_blocks.items():
                    w_indices = block_indices[(ci, cj)]
                    for k, idx in enumerate(w_indices):
                        c_row = idx // 4
                        c_col_local = idx % 4
                        c_col_global = block * 4 + c_col_local
                        w_flat_idx = c_col_global * n + c_row
                        W[w_flat_idx][col] += w_2[k] * coeff_C
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

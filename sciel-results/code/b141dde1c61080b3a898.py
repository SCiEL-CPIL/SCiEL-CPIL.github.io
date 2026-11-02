"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # u, v, w are length 4 vectors (flattened 2x2)
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
    
    # Define block mappings for 4x4 matrices
    # Block (i, j) of 4x4 matrix corresponds to indices in flattened 16-vector
    # Row-major: index = r * 4 + c
    # Block (0,0): r in 0-1, c in 0-1 -> 0, 1, 4, 5
    # Block (0,1): r in 0-1, c in 2-3 -> 2, 3, 6, 7
    # Block (1,0): r in 2-3, c in 0-1 -> 8, 9, 12, 13
    # Block (1,1): r in 2-3, c in 2-3 -> 10, 11, 14, 15
    
    def get_block_indices(block_i, block_j):
        # block_i, block_j in {0, 1}
        start_r = block_i * 2
        start_c = block_j * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = start_r + dr
                c = start_c + dc
                indices.append(r * 4 + c)
        return indices

    # Precompute indices for all 4 blocks
    block_indices = {}
    for bi in range(2):
        for bj in range(2):
            block_indices[(bi, bj)] = get_block_indices(bi, bj)

    # 4x4x4 decomposition uses 8 products of 2x2 blocks
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    products = [
        ((0, 0), (0, 0), (0, 0)),  # A_00 B_00 -> C_00
        ((0, 1), (1, 0), (0, 0)),  # A_01 B_10 -> C_00
        ((0, 0), (0, 1), (0, 1)),  # A_00 B_01 -> C_01
        ((0, 1), (1, 1), (0, 1)),  # A_01 B_11 -> C_01
        ((1, 0), (0, 0), (1, 0)),  # A_10 B_00 -> C_10
        ((1, 1), (1, 0), (1, 0)),  # A_11 B_10 -> C_10
        ((1, 0), (0, 1), (1, 1)),  # A_10 B_01 -> C_11
        ((1, 1), (1, 1), (1, 1))   # A_11 B_11 -> C_11
    ]
    
    # Each 4x4x4 block uses 8 * 7 = 56 rank-1 terms
    # 4x4x8 uses two 4x4x4 blocks, so 2 * 56 = 112 rank-1 terms
    rank = 112
    
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 112
    
    for block in range(2):
        base_col = block * 56
        for prod_idx, (a_block, b_block, c_block) in enumerate(products):
            for s_idx, (u_2, v_2, w_2) in enumerate(strassen_2x2):
                col = base_col + prod_idx * 7 + s_idx
                
                # Map u_2 to U
                u_indices = block_indices[a_block]
                for k, idx in enumerate(u_indices):
                    U[idx][col] = u_2[k]
                    
                # Map v_2 to V
                # V index: j * p + k
                # For 4x4x4 block, j in 0..3, k in 0..3
                # But V is for m*p = 4*8 = 32.
                # The second block of p is k=4..7.
                # So for block 0, k_global = k_local.
                # For block 1, k_global = 4 + k_local.
                v_indices = block_indices[b_block]
                for k, idx in enumerate(v_indices):
                    # idx is the flattened index in 4x4 B (0..15)
                    # b_row = idx // 4, b_col = idx % 4
                    # In V, the index is b_row * p + b_col_global
                    b_row = idx // 4
                    b_col_local = idx % 4
                    b_col_global = block * 4 + b_col_local
                    v_flat_idx = b_row * p + b_col_global
                    V[v_flat_idx][col] = v_2[k]
                    
                # Map w_2 to W
                # W index: k * n + i
                # For 4x4x4 block, k in 0..3, i in 0..3
                # W is for p*n = 8*4 = 32.
                # For block 0, k_global = k_local.
                # For block 1, k_global = 4 + k_local.
                w_indices = block_indices[c_block]
                for k, idx in enumerate(w_indices):
                    # idx is the flattened index in 4x4 C (0..15)
                    # c_row = idx // 4, c_col = idx % 4
                    # c_row is i (row of C, which is row of A)
                    # c_col is k (col of C, which is col of B)
                    c_row = idx // 4
                    c_col_local = idx % 4
                    c_col_global = block * 4 + c_col_local
                    w_flat_idx = c_col_global * n + c_row
                    W[w_flat_idx][col] = w_2[k]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

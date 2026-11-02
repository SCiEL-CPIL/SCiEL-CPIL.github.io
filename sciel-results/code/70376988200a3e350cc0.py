"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
    ]
    
    # Helper to get indices for a 2x2 block in a 4x4 matrix
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

    # Precompute indices for A and B blocks (4x4 layout)
    # A is 4x4, B is 4x4 (for one half)
    block_indices = {}
    for bi in range(2):
        for bj in range(2):
            block_indices[(bi, bj)] = get_block_indices(bi, bj)

    # Define the 7 Strassen linear combinations for 4x4x4
    # L_s is a linear combo of A blocks: L_s = sum c_{sb} A_b
    # R_s is a linear combo of B blocks: R_s = sum c_{sb} B_b
    # We store as list of (block_idx, coeff)
    strassen_4x4 = [
        # s=0: A00 * (B00 + B11)
        ([( (0,0), 1)], [((0,0), 1), ((1,1), 1)]),
        # s=1: (A00 + A11) * B11
        ([( (0,0), 1), ((1,1), 1)], [((1,1), 1)]),
        # s=2: (A10 + A00) * B00
        ([( (1,0), 1), ((0,0), 1)], [((0,0), 1)]),
        # s=3: A11 * (B10 + B00)
        ([( (1,1), 1)], [((1,0), 1), ((0,0), 1)]),
        # s=4: (A01 + A11) * B10
        ([( (0,1), 1), ((1,1), 1)], [((1,0), 1)]),
        # s=5: (A10 + A11) * (B01 - B00)
        ([( (1,0), 1), ((1,1), 1)], [((0,1), 1), ((0,0), -1)]),
        # s=6: (A00 + A01) * (B10 + B11)
        ([( (0,0), 1), ((0,1), 1)], [((1,0), 1), ((1,1), 1)])
    ]

    # We need to accumulate U, V, W
    # U: 16 x Rank
    # V: 32 x Rank
    # W: 32 x Rank
    # Rank = 2 * 7 * 7 = 98
    
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    for half in range(2):
        # half 0: B columns 0-3, C columns 0-3
        # half 1: B columns 4-7, C columns 4-7
        
        for s_idx, (L_terms, R_terms) in enumerate(strassen_4x4):
            # For each of the 7 scalar Strassen terms
            for j_idx, (u_vec, v_vec, w_vec) in enumerate(strassen_2x2):
                col = col_offset + s_idx * 7 + j_idx
                
                # Add to U
                # L_terms is list of ((bi, bj), coeff)
                for (bi, bj), coeff_L in L_terms:
                    # u_vec is flattened 2x2, indices 0..3 correspond to (0,0),(0,1),(1,0),(1,1) of the 2x2 block
                    # We need to map u_vec[k] to U[global_idx][col]
                    # global_idx is the index in the 4x4 A matrix for the element (dr+bi*2, dc+bj*2)
                    block_idx_list = block_indices[(bi, bj)]
                    for k in range(4):
                        # k is index in 2x2 block
                        # dr = k // 2, dc = k % 2
                        dr = k // 2
                        dc = k % 2
                        # global row = bi*2 + dr
                        # global col = bj*2 + dc
                        # global flat index = (bi*2 + dr) * 4 + (bj*2 + dc)
                        # This is exactly block_idx_list[k]
                        u_val = u_vec[k] * coeff_L
                        if u_val != 0:
                            U[block_idx_list[k]][col] += u_val

                # Add to V
                # V index: j * p + k_global
                # j is row of B (0..3), k_global is col of B (0..7)
                # R_terms is list of ((bi, bj), coeff)
                # bi, bj are block indices in 4x4 B.
                # Row of B element = bi*2 + dr
                # Col of B element (local) = bj*2 + dc
                # Col of B element (global) = half*4 + bj*2 + dc
                for (bi, bj), coeff_R in R_terms:
                    block_idx_list = block_indices[(bi, bj)]
                    for k in range(4):
                        dr = k // 2
                        dc = k % 2
                        b_row = bi * 2 + dr
                        b_col_local = bj * 2 + dc
                        b_col_global = half * 4 + b_col_local
                        v_flat_idx = b_row * p + b_col_global
                        v_val = v_vec[k] * coeff_R
                        if v_val != 0:
                            V[v_flat_idx][col] += v_val

                # Add to W
                # W index: k * n + i
                # k is col of C (0..7), i is row of C (0..3)
                # w_vec corresponds to the 2x2 block of C.
                # The product M_s = L_s R_s results in a 2x2 matrix.
                # This 2x2 matrix corresponds to a specific block of C?
                # NO. The Strassen formula for 4x4x4 says:
                # C_00 = M_0 + M_3 - M_1 + M_5
                # C_01 = M_4 + M_6
                # C_10 = M_2 + M_3
                # C_11 = M_0 - M_2 + M_4 + M_6
                # Wait, I used indices 0..6 in strassen_4x4.
                # Let's check the mapping of M_s to C blocks.
                # M_s is a 2x2 matrix.
                # We need to add w_vec to the appropriate block of C.
                # The contribution of M_s to C_block is coeff * M_s.
                # So for each s, we need to know which C block it adds to and with what sign.
                
                # Let's define the mapping from s to C blocks.
                # s=0 (M0): C00 (+1), C11 (+1)
                # s=1 (M1): C00 (-1)
                # s=2 (M2): C10 (+1), C11 (-1)
                # s=3 (M3): C00 (+1), C10 (+1)
                # s=4 (M4): C01 (+1), C11 (+1)
                # s=5 (M5): C00 (+1)
                # s=6 (M6): C01 (+1), C11 (+1)
                
                # Wait, let's re-verify the standard Strassen output formula.
                # C00 = M1 + M4 - M2 + M6 (using 1-based M1..M7)
                # C01 = M5 + M7
                # C10 = M3 + M4
                # C11 = M1 - M3 + M5 + M7
                # My strassen_4x4 list order:
                # s=0: M1
                # s=1: M2
                # s=2: M3
                # s=3: M4
                # s=4: M5
                # s=5: M6
                # s=6: M7
                
                # Mapping:
                # s=0 (M1): C00 (+1), C11 (+1)
                # s=1 (M2): C00 (-1)
                # s=2 (M3): C10 (+1), C11 (-1)
                # s=3 (M4): C00 (+1), C10 (+1)
                # s=4 (M5): C01 (+1), C11 (+1)
                # s=5 (M6): C00 (+1)
                # s=6 (M7): C01 (+1), C11 (+1)
                
                c_block_map = {
                    0: [((0,0), 1), ((1,1), 1)],
                    1: [((0,0), -1)],
                    2: [((1,0), 1), ((1,1), -1)],
                    3: [((0,0), 1), ((1,0), 1)],
                    4: [((0,1), 1), ((1,1), 1)],
                    5: [((0,0), 1)],
                    6: [((0,1), 1), ((1,1), 1)]
                }
                
                for (ci, cj), coeff_C in c_block_map[s_idx]:
                    # ci, cj is the block index in C (4x4)
                    # w_vec is flattened 2x2.
                    # C element at (ci*2+dr, cj*2+dc)
                    # W index: k * n + i
                    # k = col of C = cj*2 + dc
                    # i = row of C = ci*2 + dr
                    # k_global = half*4 + cj*2 + dc
                    block_idx_list = block_indices[(ci, cj)]
                    for k in range(4):
                        dr = k // 2
                        dc = k % 2
                        c_row = ci * 2 + dr
                        c_col_local = cj * 2 + dc
                        c_col_global = half * 4 + c_col_local
                        w_flat_idx = c_col_global * n + c_row
                        w_val = w_vec[k] * coeff_C
                        if w_val != 0:
                            W[w_flat_idx][col] += w_val
                            
        col_offset += 49
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

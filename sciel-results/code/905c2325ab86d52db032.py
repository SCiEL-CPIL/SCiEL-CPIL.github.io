"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices 0..4 into blocks of sizes 2, 2, 1
    # Block 0: {0, 1}
    # Block 1: {2, 3}
    # Block 2: {4}
    block_sizes = [2, 2, 1]
    block_offsets = [0, 2, 4]
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # We will build lists of columns for U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a local decomposition to the global factors
    # local_U: list of lists (size_a * size_b) x rank
    # local_V: list of lists (size_b * size_c) x rank
    # local_W: list of lists (size_c * size_a) x rank -- Note: W index is k*n+i, so row is k, col is i.
    # In our local C, row is i, col is k.
    # So local_W[i, k] corresponds to global W[k*n + i].
    def add_block_decomposition(I, J, K, local_U, local_V, local_W):
        size_I = block_sizes[I]
        size_J = block_sizes[J]
        size_K = block_sizes[K]
        
        off_I = block_offsets[I]
        off_J = block_offsets[J]
        off_K = block_offsets[K]
        
        rank = len(local_U[0])
        
        for t in range(rank):
            u_col = [0] * U_rows
            v_col = [0] * V_rows
            w_col = [0] * W_rows
            
            # Fill U
            for li in range(size_I):
                for lj in range(size_J):
                    val = local_U[li][lj][t]
                    if val != 0:
                        gi = off_I + li
                        gj = off_J + lj
                        # A is n x m. Index i*m + j.
                        u_col[gi * m + gj] = val
            
            # Fill V
            for lj in range(size_J):
                for lk in range(size_K):
                    val = local_V[lj][lk][t]
                    if val != 0:
                        gj = off_J + lj
                        gk = off_K + lk
                        # B is m x p. Index j*p + k.
                        v_col[gj * p + gk] = val
            
            # Fill W
            for lk in range(size_K):
                for li in range(size_I):
                    val = local_W[li][lk][t] # local_W is indexed [i, k]
                    if val != 0:
                        gk = off_K + lk
                        gi = off_I + li
                        # C is p x n? No, C is n x p (5x5).
                        # Task says W[k*n+i, r].
                        # So row index in W is k*n + i.
                        w_col[gk * n + gi] = val
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # Define local decompositions
    
    # Case 1: 2x2 * 2x2 -> 2x2 (Rank 7, Strassen)
    # local_U[i, j], local_V[j, k], local_W[i, k]
    U_22 = [
        [[1,0,0,0,0,0,0], [0,0,0,0,1,0,1], [0,0,0,0,0,1,0], [0,0,0,1,0,0,-1]],
        # t=0: a00, a11. t=1: a10, a11. t=2: a00. t=3: a11. t=4: a00, a01. t=5: a10, -a00. t=6: a01, -a11
        # Wait, let's list by t
    ]
    # Let's construct U_22 as list of columns
    U_22_cols = [
        [1, 0, 0, 1], # t0: a00, a11
        [0, 0, 1, 1], # t1: a10, a11
        [1, 0, 0, 0], # t2: a00
        [0, 0, 0, 1], # t3: a11
        [1, 1, 0, 0], # t4: a00, a01
        [-1, 0, 1, 0],# t5: -a00, a10
        [0, 1, 0, -1] # t6: a01, -a11
    ]
    # Flat index: i*2 + j.
    # t0: 0, 3
    # t1: 2, 3
    # t2: 0
    # t3: 3
    # t4: 0, 1
    # t5: 0, 2
    # t6: 1, 3
    
    V_22_cols = [
        [1, 0, 0, 1], # t0: b00, b11
        [1, 0, 0, 0], # t1: b00
        [0, 1, 0, -1],# t2: b01, -b11
        [-1, 0, 1, 0],# t3: -b00, b10
        [0, 0, 0, 1], # t4: b11
        [1, 1, 0, 0], # t5: b00, b01
        [0, 0, 1, 1]  # t6: b10, b11
    ]
    
    W_22_cols = [
        [1, 0, 0, 1], # t0: c00, c11
        [0, 0, 1, -1],# t1: c10, -c11
        [0, 1, 0, 1], # t2: c01, c11
        [1, 0, 1, 0], # t3: c00, c10
        [-1, 1, 0, 0],# t4: -c00, c01
        [0, 0, 0, 1], # t5: c11
        [1, 0, 0, 1]  # t6: c00, c11
    ]
    # Flat index for W: i*2 + k? No, W is [i, k].
    # c00 (0,0), c01 (0,1), c10 (1,0), c11 (1,1).
    # Flat: 0, 1, 2, 3.
    # t0: c00, c11 -> 0, 3
    # t1: c10, -c11 -> 2, 3
    # t2: c01, c11 -> 1, 3
    # t3: c00, c10 -> 0, 2
    # t4: -c00, c01 -> 0, 1
    # t5: c11 -> 3
    # t6: c00, c11 -> 0, 3
    
    # We need to convert these flat columns to 2D lists [i][j] for the helper
    def flat_to_2d(flat_cols, rows, cols):
        res = [[0]*len(flat_cols) for _ in range(rows*cols)]
        for t in range(len(flat_cols)):
            for idx, val in enumerate(flat_cols[t]):
                if val != 0:
                    r = idx // cols
                    c = idx % cols
                    res[r*cols + c][t] = val
        return res

    U_22_2d = flat_to_2d(U_22_cols, 2, 2)
    V_22_2d = flat_to_2d(V_22_cols, 2, 2)
    W_22_2d = flat_to_2d(W_22_cols, 2, 2)
    
    # Case 2: 2x2 * 2x1 -> 2x1 (Rank 4)
    # U (2x2), V (2x1), W (2x1)
    U_221_cols = [
        [1, 0, 0, 0], # a00
        [0, 1, 0, 0], # a01
        [0, 0, 1, 0], # a10
        [0, 0, 0, 1]  # a11
    ]
    V_221_cols = [
        [1, 0], # v0
        [0, 1], # v1
        [1, 0], # v0
        [0, 1]  # v1
    ]
    W_221_cols = [
        [1, 0], # c0
        [1, 0], # c0
        [0, 1], # c1
        [0, 1]  # c1
    ]
    U_221_2d = flat_to_2d(U_221_cols, 2, 2)
    V_221_2d = flat_to_2d(V_221_cols, 2, 1)
    W_221_2d = flat_to_2d(W_221_cols, 2, 1)
    
    # Case 3: 2x1 * 1x2 -> 2x2 (Rank 4)
    U_212_cols = [
        [1, 0], # a0
        [1, 0], # a0
        [0, 1], # a1
        [0, 1]  # a1
    ]
    V_212_cols = [
        [1, 0], # b0
        [0, 1], # b1
        [1, 0], # b0
        [0, 1]  # b1
    ]
    W_212_cols = [
        [1, 0, 0, 0], # c00
        [0, 1, 0, 0], # c01
        [0, 0, 1, 0], # c10
        [0, 0, 0, 1]  # c11
    ]
    U_212_2d = flat_to_2d(U_212_cols, 2, 1)
    V_212_2d = flat_to_2d(V_212_cols, 1, 2)
    W_212_2d = flat_to_2d(W_212_cols, 2, 2)
    
    # Case 4: 1x2 * 2x2 -> 1x2 (Rank 4)
    U_122_cols = [
        [1, 0], # a0
        [0, 1], # a1
        [1, 0], # a0
        [0, 1]  # a1
    ]
    V_122_cols = [
        [1, 0, 0, 0], # b00
        [0, 0, 1, 0], # b10
        [0, 1, 0, 0], # b01
        [0, 0, 0, 1]  # b11
    ]
    W_122_cols = [
        [1, 0], # c0
        [1, 0], # c0
        [0, 1], # c1
        [0, 1]  # c1
    ]
    U_122_2d = flat_to_2d(U_122_cols, 1, 2)
    V_122_2d = flat_to_2d(V_122_cols, 2, 2)
    W_122_2d = flat_to_2d(W_122_cols, 1, 2)
    
    # Case 5: 1x2 * 2x1 -> 1x1 (Rank 2)
    U_121_cols = [
        [1, 0], # a0
        [0, 1]  # a1
    ]
    V_121_cols = [
        [1, 0], # v0
        [0, 1]  # v1
    ]
    W_121_cols = [
        [1], # c
        [1]  # c
    ]
    U_121_2d = flat_to_2d(U_121_cols, 1, 2)
    V_121_2d = flat_to_2d(V_121_cols, 2, 1)
    W_121_2d = flat_to_2d(W_121_cols, 1, 1)
    
    # Case 6: 1x1 * 1x2 -> 1x2 (Rank 2)
    U_112_cols = [
        [1], # a
        [1]  # a
    ]
    V_112_cols = [
        [1, 0], # b0
        [0, 1]  # b1
    ]
    W_112_cols = [
        [1, 0], # c0
        [0, 1]  # c1
    ]
    U_112_2d = flat_to_2d(U_112_cols, 1, 1)
    V_112_2d = flat_to_2d(V_112_cols, 1, 2)
    W_112_2d = flat_to_2d(W_112_cols, 1, 2)
    
    # Case 7: 2x1 * 1x1 -> 2x1 (Rank 2)
    U_211_cols = [
        [1, 0], # a0
        [0, 1]  # a1
    ]
    V_211_cols = [
        [1], # b
        [1]  # b
    ]
    W_211_cols = [
        [1, 0], # c0
        [0, 1]  # c1
    ]
    U_211_2d = flat_to_2d(U_211_cols, 2, 1)
    V_211_2d = flat_to_2d(V_211_cols, 1, 1)
    W_211_2d = flat_to_2d(W_211_cols, 2, 1)
    
    # Case 8: 1x1 * 1x1 -> 1x1 (Rank 1)
    U_111_cols = [
        [1]
    ]
    V_111_cols = [
        [1]
    ]
    W_111_cols = [
        [1]
    ]
    U_111_2d = flat_to_2d(U_111_cols, 1, 1)
    V_111_2d = flat_to_2d(V_111_cols, 1, 1)
    W_111_2d = flat_to_2d(W_111_cols, 1, 1)
    
    # Iterate over blocks
    for I in range(3):
        for J in range(3):
            for K in range(3):
                si = block_sizes[I]
                sj = block_sizes[J]
                sk = block_sizes[K]
                
                if si == 2 and sj == 2 and sk == 2:
                    add_block_decomposition(I, J, K, U_22_2d, V_22_2d, W_22_2d)
                elif si == 2 and sj == 2 and sk == 1:
                    add_block_decomposition(I, J, K, U_221_2d, V_221_2d, W_221_2d)
                elif si == 2 and sj == 1 and sk == 2:
                    add_block_decomposition(I, J, K, U_212_2d, V_212_2d, W_212_2d)
                elif si == 1 and sj == 2 and sk == 2:
                    add_block_decomposition(I, J, K, U_122_2d, V_122_2d, W_122_2d)
                elif si == 1 and sj == 2 and sk == 1:
                    add_block_decomposition(I, J, K, U_121_2d, V_121_2d, W_121_2d)
                elif si == 1 and sj == 1 and sk == 2:
                    add_block_decomposition(I, J, K, U_112_2d, V_112_2d, W_112_2d)
                elif si == 2 and sj == 1 and sk == 1:
                    add_block_decomposition(I, J, K, U_211_2d, V_211_2d, W_211_2d)
                elif si == 1 and sj == 1 and sk == 1:
                    add_block_decomposition(I, J, K, U_111_2d, V_111_2d, W_111_2d)
                else:
                    # Should not happen for 2,2,1 partition
                    pass

    # Convert lists of columns to matrices (rows x rank)
    rank = len(U_cols)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for t in range(rank):
        for i in range(U_rows):
            U[i][t] = U_cols[t][i]
        for i in range(V_rows):
            V[i][t] = V_cols[t][i]
        for i in range(W_rows):
            W[i][t] = W_cols[t][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

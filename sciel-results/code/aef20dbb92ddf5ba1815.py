# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U: A, V: B, W: C
    # Indices 0:00, 1:01, 2:10, 3:11
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),      # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),      # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),      # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),      # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),      # M7
    ]
    
    # 4 blocks
    for b in range(4):
        # Determine global offsets
        # A is 2x4. Blocks:
        # b=0: A[0:2, 0:2] -> A_L, B_LL
        # b=1: A[0:2, 2:4] -> A_R, B_RL ? No.
        # Let's re-verify the block mapping from the previous code.
        # Previous code:
        # Block 0: A_L (j=0,1) * B_LL (j=0,1, r=0,1) -> C_L (r=0,1)
        # Block 1: A_R (j=2,3) * B_RL (j=2,3, r=0,1) -> C_L (r=0,1)
        # Block 2: A_L (j=0,1) * B_LR (j=0,1, r=2,3) -> C_R (r=2,3)
        # Block 3: A_R (j=2,3) * B_RR (j=2,3, r=2,3) -> C_R (r=2,3)
        
        # This mapping covers all combinations of A_col_block and B_col_block.
        # A_col_block: 0 (cols 0-1), 1 (cols 2-3)
        # B_col_block: 0 (cols 0-1), 1 (cols 2-3)
        # b=0: A_col 0, B_col 0
        # b=1: A_col 1, B_col 0
        # b=2: A_col 0, B_col 1
        # b=3: A_col 1, B_col 1
        
        a_col_offset = 0 if b in [0, 2] else 2
        b_col_offset = 0 if b in [0, 1] else 2
        c_col_offset = 0 if b in [0, 1] else 2
        
        for s in range(7):
            col = b * 7 + s
            u_c, v_c, w_c = terms[s]
            
            for idx, coeff in u_c.items():
                li = idx // 2
                lj = idx % 2
                # A global index: i * 4 + j
                # i is row (0,1), j is col (0..3)
                # li is local row, lj is local col (0,1)
                g_i = li
                g_j = a_col_offset + lj
                g_idx = g_i * 4 + g_j
                U[g_idx][col] += coeff
                
            for idx, coeff in v_c.items():
                li = idx // 2
                lj = idx % 2
                # B global index: j * 4 + r
                # j is row (0..3), r is col (0..3)
                # In our block, B is 2x2.
                # Local row (j_local) maps to global j.
                # Local col (r_local) maps to global r.
                # The term V corresponds to B.
                # In Strassen, V is the second factor.
                # So V indices are (j, r).
                # li is j_local, lj is r_local.
                g_j = b_col_offset + li
                g_r = lj
                g_idx = g_j * 4 + g_r
                V[g_idx][col] += coeff
                
            for idx, coeff in w_c.items():
                li = idx // 2
                lj = idx % 2
                # C global index: r * 2 + i
                # i is row (0,1), r is col (0..3)
                # li is i_local, lj is r_local.
                g_i = li
                g_r = c_col_offset + lj
                g_idx = g_r * 2 + g_i
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

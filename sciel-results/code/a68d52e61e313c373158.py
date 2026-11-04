"""Combined: explicit Strassen 4x4x4 (49) + 4x(2x2x3) (44) construction for 4x4x7."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        return np.max(np.abs(result - 1.0))

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Explicit construction: rank 49 (Strassen 4x4x4) + 44 (4 * 2x2x3) ----
    total_rank = 93
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    current_rank = 0

    # Strassen 2x2x2: 7 terms. Each term = (A_sub, B_sub, C_sub)
    # A_sub, B_sub are 2x2 sparse dicts {(row,col): coeff}
    # C_sub: which output entry each term contributes to
    strassen_2x2x2 = [
        # term: A, B, C (list of (row,col,coeff))
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, [(0,0,1),(1,1,1)]),   # M1
        ({(1,0):1,(1,1):1}, {(0,0):1}, [(1,0,1),(1,1,-1)]),           # M2
        ({(0,0):1}, {(0,1):1,(1,1):-1}, [(0,1,1),(1,1,1)]),           # M3
        ({(1,1):1}, {(1,0):1,(0,0):-1}, [(0,0,1),(1,0,1)]),           # M4
        ({(0,0):1,(0,1):1}, {(1,1):1}, [(0,0,-1),(0,1,1)]),           # M5
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, [(1,1,1)]),           # M6
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, [(0,0,1)]),           # M7
    ]

    def add_strassen_block(a_row, a_col, b_row, b_col, c_row, c_col):
        """Add a Strassen 2x2x2 decomposition for a 2x2 A block, 2x2 B block, 2x2 C block.
        A rows [a_row, a_row+1], cols [a_col, a_col+1]
        B rows [b_row, b_row+1], cols [b_col, b_col+1]
        C rows [c_row, c_row+1], cols [c_col, c_col+1]
        Returns number of rank terms added."""
        nonlocal current_rank
        for t_idx in range(7):
            A_sub, B_sub, C_sub = strassen_2x2x2[t_idx]
            r = current_rank
            current_rank += 1
            # Fill U (A)
            for (li, lj), c in A_sub.items():
                i = a_row + li
                j = a_col + lj
                U[i * m + j, r] += c
            # Fill V (B)
            for (li, lj), c in B_sub.items():
                i = b_row + li
                j = b_col + lj
                V[i * p + j, r] += c
            # Fill W (C)
            for (li, lj, c) in C_sub:
                i = c_row + li
                j = c_col + lj
                W[j * n + i, r] += c
        return 7

    # ---- Part 1: 4x4x4 via two-level Strassen (rank 49) ----
    # Top level: 7 terms combining 2x2 blocks of A (4 blocks), B (4 blocks), C (4 blocks)
    # A blocks: (0,0)=rows0-1,cols0-1 ; (0,1)=rows0-1,cols2-3 ; (1,0)=rows2-3,cols0-1 ; (1,1)=rows2-3,cols2-3
    # B blocks: (0,0)=rows0-1,cols0-1 ; (0,1)=rows0-1,cols2-3 ; (1,0)=rows2-3,cols0-1 ; (1,1)=rows2-3,cols2-3
    # C blocks: (0,0)=rows0-1,cols0-1 ; (0,1)=rows0-1,cols2-3 ; (1,0)=rows2-3,cols0-1 ; (1,1)=rows2-3,cols2-3
    #
    # Top-level Strassen terms (M1..M7) define combinations of A-blocks and B-blocks.
    # Then each top term is a 2x2x2 Strassen over the 4 sub-blocks.
    #
    # Top-level: A_block_combos and B_block_combos for M1..M7
    top_terms = [
        # (A_block_dict, B_block_dict, C_block_dict)
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),   # M1
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),           # M2
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),           # M3
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),           # M4
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),           # M5
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),           # M6
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),           # M7
    ]

    for m_idx in range(7):
        A_blocks, B_blocks, C_blocks = top_terms[m_idx]
        for t_idx in range(7):
            A_sub, B_sub, C_sub = strassen_2x2x2[t_idx]
            r = current_rank
            current_rank += 1
            # U: combine A_block coefficients with A_sub
            for (li, lj), c_sub in A_sub.items():
                for (br, bc), c_blk in A_blocks.items():
                    i = 2 * br + li
                    j = 2 * bc + lj
                    U[i * m + j, r] += c_sub * c_blk
            # V: combine B_block coefficients with B_sub
            for (li, lj), c_sub in B_sub.items():
                for (br, bc), c_blk in B_blocks.items():
                    i = 2 * br + li
                    j = 2 * bc + lj
                    V[i * p + j, r] += c_sub * c_blk
            # W: combine C_block coefficients with C_sub
            for (li, lj, c_sub) in C_sub:
                for (br, bc), c_blk in C_blocks.items():
                    i = 2 * br + li
                    j = 2 * bc + lj
                    W[j * n + i, r] += c_sub * c_blk
    # current_rank should be 49 now
    assert current_rank == 49, f"Expected 49, got {current_rank}"

    # ---- Part 2: 4x4x3 (cols 4,5,6 of B) via 4 * (2x2x3) (rank 44) ----
    # For each 2x2 A block, multiply with the corresponding 2x3 B block (cols 4-6)
    # to produce a 2x3 C block.
    #
    # A_00 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 4-6) -> C_top (rows 0-1, cols 4-6)
    # A_01 (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 4-6) -> C_top (rows 0-1, cols 4-6)
    # A_10 (rows 2-3, cols 0-1) * B_top (rows 0-1, cols 4-6) -> C_bot (rows 2-3, cols 4-6)
    # A_11 (rows 2-3, cols 2-3) * B_bot (rows 2-3, cols 4-6) -> C_bot (rows 2-3, cols 4-6)

    def add_2x2x3(a_row, a_col, b_row, c_row):
        """Add a 2x2x3 decomposition (rank 11) for A 2x2 block, B 2x3 block, C 2x3 block.
        A: rows [a_row, a_row+1], cols [a_col, a_col+1]
        B: rows [b_row, b_row+1], cols [4, 5, 6]
        C: rows [c_row, c_row+1], cols [4, 5, 6]
        """
        nonlocal current_rank
        # Strassen part: cols 4,5 (2 columns) -> 7 terms
        for t_idx in range(7):
            A_sub, B_sub, C_sub = strassen_2x2x2[t_idx]
            r = current_rank
            current_rank += 1
            # U: A entries
            for (li, lj), c in A_sub.items():
                i = a_row + li
                j = a_col + lj
                U[i * m + j, r] += c
            # V: B entries (cols 4, 5)
            for (li, lj), c in B_sub.items():
                i = b_row + li
                j = 4 + lj
                V[i * p + j, r] += c
            # W: C entries (cols 4, 5)
            for (li, lj, c) in C_sub:
                i = c_row + li
                j = 4 + lj
                W[j * n + i, r] += c
        # Schoolbook part: col 6 (1 column) -> 4 terms
        for li in range(2):
            for lj in range(2):
                r = current_rank
                current_rank += 1
                i = a_row + li
                j = a_col + lj
                U[i * m + j, r] = 1
                i_b = b_row + li
                V[i_b * p + 6, r] = 1
                i_c = c_row + li
                W[6 * n + i_c, r] = 1
        return 11

    # A_00 * B_top -> C_top
    add_2x2x3(a_row=0, a_col=0, b_row=0, c_row=0)
    # A_01 * B_bot -> C_top
    add_2x2x3(a_row=0, a_col=2, b_row=2, c_row=0)
    # A_10 * B_top -> C_bot
    add_2x2x3(a_row=2, a_col=0, b_row=0, c_row=2)
    # A_11 * B_bot -> C_bot
    add_2x2x3(a_row=2, a_col=2, b_row=2, c_row=2)

    assert current_rank == 93, f"Expected 93, got {current_rank}"

    # Verify the construction
    err = verify(U.tolist(), V.tolist(), W.tolist(), 93)
    if err < 1e-9:
        # Convert to lists (entries are integers, so exact)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        # Round to nearest integer for cleanliness (they should already be integers)
        U_round = np.round(U).astype(int).tolist()
        V_round = np.round(V).astype(int).tolist()
        W_round = np.round(W).astype(int).tolist()
        err_round = verify(U_round, V_round, W_round, 93)
        if err_round < 1e-9:
            return {"U": U_round, "V": V_round, "W": W_round}
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: if construction failed verification, use schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END

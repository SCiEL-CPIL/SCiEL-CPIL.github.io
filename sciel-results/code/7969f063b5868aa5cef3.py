import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 5

    def time_remaining():
        return deadline - time.time()

    def build_target_tensor():
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T

    TARGET_T = build_target_tensor()

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False

        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False

        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)

        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

        if not np.allclose(T, TARGET_T, atol=1e-6):
            return False
        return True

    def get_schoolbook():
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Rank-65 decomposition: 49 Strassen terms for 4x4x4 + 16 for 5th column
    U = np.zeros((n * m, 65), dtype=float)
    V = np.zeros((m * p, 65), dtype=float)
    W = np.zeros((p * n, 65), dtype=float)

    # Strassen 2x2 base
    strassen_mults = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]

    inner_contrib = {
        0: [(0,0, 1), (1,1, 1)],
        1: [(1,0, 1), (1,1, -1)],
        2: [(0,1, 1), (1,1, 1)],
        3: [(0,0, 1), (1,0, 1)],
        4: [(0,0, -1), (0,1, 1)],
        5: [(1,1, 1)],
        6: [(0,0, 1)],
    }

    block_offsets = {
        'C00': (0, 0),
        'C01': (0, 2),
        'C10': (2, 0),
        'C11': (2, 2),
    }

    outer_block_map = {
        0: [('C00', 1), ('C11', 1)],
        1: [('C10', 1), ('C11', -1)],
        2: [('C01', 1), ('C11', 1)],
        3: [('C00', 1), ('C10', 1)],
        4: [('C00', -1), ('C01', 1)],
        5: [('C11', 1)],
        6: [('C00', 1)],
    }

    outer_terms = [
        ([[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]],
         [[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]]),
        ([[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]],
         [[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]]),
        ([[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]],
         [[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]]),
        ([[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]],
         [[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]]),
        ([[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]],
         [[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]]),
        ([[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]],
         [[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]]),
        ([[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]],
         [[(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)],
          [(0,0,1),(0,2,1),(1,1,1),(1,3,1)], [(0,0,1),(0,2,1),(1,1,1),(1,3,1)]]),
    ]

    # Build the actual Strassen decomposition properly
    # Using the standard recursive approach

    def build_strassen_4x4x4():
        """Build rank-49 decomposition for 4x4x4 matrix multiplication."""
        U4 = np.zeros((n * m, 49), dtype=float)
        V4 = np.zeros((m * 4, 49), dtype=float)
        W4 = np.zeros((4 * n, 49), dtype=float)

        # Outer Strassen: 7 terms, each is a 2x2 multiplication
        # Each 2x2 multiplication uses inner Strassen: 7 terms
        # Total: 7 * 7 = 49 terms

        # Define the 7 outer Strassen patterns for 4x4 blocks
        # Each pattern defines how to form the U and V factors for a 2x2 block multiplication
        # The 2x2 blocks are indexed by (block_row, block_col) where block_row, block_col in {0,1}
        # Within each block, positions are (0,0), (0,1), (1,0), (1,1)
        # Global position = (block_idx * 2 + local_idx)

        # Outer Strassen patterns (for the 2x2 block structure):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)

        # For each outer term, define which 2x2 blocks of A and B contribute
        outer_patterns = [
            # (A_blocks, B_blocks) where each is a dict: (block_r, block_c) -> coeff
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),              # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),              # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),   # M7
        ]

        # For each outer term, the 2x2 block multiplication is done via inner Strassen
        # The inner Strassen operates on the 2x2 scalar entries within each block
        # But since we're at the block level, we need to expand each 2x2 block multiplication
        # into 7 scalar multiplications using inner Strassen

        # For a 2x2 block multiplication C = A_block * B_block:
        # The inner Strassen gives 7 scalar multiplications
        # Each scalar multiplication involves specific entries of A_block and B_block
        # A_block entries: A[br*2+lr, bc*2+lc] for br, bc in {0,1} (block indices) and lr, lc in {0,1} (local)
        # B_block entries: B[br*2+lr, bc*2+lc]

        # Inner Strassen patterns for 2x2 scalar multiplication:
        inner_patterns = [
            # (A_local, B_local) where each is a dict: (lr, lc) -> coeff
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # m1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # m2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),              # m3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),              # m4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # m5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # m6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),   # m7
        ]

        # For each outer term j and inner term t, we have one scalar multiplication
        # The U factor involves specific (i,j) entries of A
        # The V factor involves specific (j,k) entries of B (where k < 4 for the 4x4x4 part)

        col = 0
        for j in range(7):
            A_blocks, B_blocks = outer_patterns[j]
            for t in range(7):
                A_local, B_local = inner_patterns[t]

                # Build U factor: for each (br, bc) in A_blocks and (lr, lc) in A_local
                for (br, bc), coeff_ab in A_blocks.items():
                    for (lr, lc), coeff_al in A_local.items():
                        i = br * 2 + lr
                        jj = bc * 2 + lc
                        U4[i * m + jj, col] += coeff_ab * coeff_al

                # Build V factor: for each (br, bc) in B_blocks and (lr, lc) in B_local
                for (br, bc), coeff_bb in B_blocks.items():
                    for (lr, lc), coeff_bl in B_local.items():
                        jj = br * 2 + lr
                        k = bc * 2 + lc
                        V4[jj * 4 + k, col] += coeff_bb * coeff_bl

                # Build W factor: the output C = sum_j,t (outer_j * inner_t)
                # For each outer term j, the result contributes to specific 2x2 blocks of C
                # For each inner term t, the result contributes to specific scalar entries within those blocks
                #
                # C = sum over outer terms of (A_outer * B_outer)
                # Each outer term produces a 2x2 block result
                # The 2x2 block result is computed via inner Strassen
                #
                # For outer term j, the 2x2 block result R_j has entries:
                # R_j[0,0] = m1 + m4 - m5 + m7
                # R_j[0,1] = m3 + m5
                # R_j[1,0] = m2 + m4
                # R_j[1,1] = m1 - m2 + m3 + m6
                #
                # So for inner term t, the contribution to R_j is:
                inner_to_block = {
                    0: [(0,0, 1), (1,1, 1)],      # m1 -> C00 + C11
                    1: [(1,0, 1), (1,1, -1)],     # m2 -> C10 - C11
                    2: [(0,1, 1), (1,1, 1)],      # m3 -> C01 + C11
                    3: [(0,0, 1), (1,0, 1)],      # m4 -> C00 + C10
                    4: [(0,0, -1), (0,1, 1)],     # m5 -> -C00 + C01
                    5: [(1,1, 1)],                # m6 -> C11
                    6: [(0,0, 1)],                # m7 -> C00
                }

                # For outer term j, which 2x2 blocks of the final C does it contribute to?
                # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
                # M2 = (A10+A11)B00 -> contributes to C10 and C11
                # M3 = A00(B01-B11) -> contributes to C01 and C11
                # M4 = A11(B10-B00) -> contributes to C00 and C10
                # M5 = (A00+A01)B11 -> contributes to C00 and C01
                # M6 = (A10-A00)(B00+B01) -> contributes to C11
                # M7 = (A01-A11)(B10+B11) -> contributes to C00

                outer_to_blocks = {
                    0: [(0,0, 1), (1,1, 1)],   # M1 -> C00 + C11
                    1: [(1,0, 1), (1,1, -1)],  # M2 -> C10 - C11
                    2: [(0,1, 1), (1,1, 1)],   # M3 -> C01 + C11
                    3: [(0,0, 1), (1,0, 1)],   # M4 -> C00 + C10
                    4: [(0,0, -1), (0,1, 1)],  # M5 -> -C00 + C01
                    5: [(1,1, 1)],             # M6 -> C11
                    6: [(0,0, 1)],             # M7 -> C00
                }

                for (cbr, cbc), coeff_ob in outer_to_blocks[j]:
                    for (clr, clc), coeff_ib in inner_to_block[t]:
                        ci = cbr * 2 + clr
                        ck = cbc * 2 + clc
                        w_idx = ck * n + ci
                        W4[w_idx, col] += coeff_ob * coeff_ib

                col += 1

        return U4, V4, W4

    U4, V4, W4 = build_strassen_4x4x4()

    # Copy the 4x4x4 part into the full 4x4x5 decomposition
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4

    # Add the 16 scalar terms for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j, 49 + i * m + j] = 1.0
            V[j * 4 + 4, 49 + i * m + j] = 1.0
            W[4 * n + i, 49 + i * m + j] = 1.0

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

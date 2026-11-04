"""Combined program: deterministic Strassen hybrid + verification + light optimization."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute index arrays for verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify_fast(Ua, Va, Wa):
        """Fast verification. Returns max absolute error."""
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        return np.max(np.abs(result - 1.0))

    def build_hybrid_decomposition():
        """
        Build a hybrid decomposition using Strassen for 4x4x4 (rank 49)
        and a 2x2x3-based approach for the remaining 4x4x3 part.
        """
        total_rank = 93  # 49 + 44
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        current_rank = 0

        # Strassen 2x2x2 terms: (U_coeffs, V_coeffs)
        # Each term: {index_in_2x2: coeff}
        s22 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ]

        # C matrix coefficients for 2x2x2 Strassen:
        # C[0,0] = M1 - M5 + M7, C[0,1] = M3 + M5
        # C[1,0] = M2 + M4, C[1,1] = M1 - M2 + M3 + M6
        c22 = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1},
            {(0,0): -1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1},
        ]

        # Part 1: 4x4x4 using two levels of Strassen (rank 49)
        # Outer Strassen on 2x2 blocks of 4x4 matrices
        # Inner Strassen on 2x2 blocks within each 2x2 block
        s44_outer = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ]

        c44_outer = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1},
            {(0,0): -1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1},
        ]

        for outer_idx in range(7):
            outer_x, outer_y = s44_outer[outer_idx]
            for inner_idx in range(7):
                r = current_rank
                current_rank += 1
                inner_x, inner_y = s22[inner_idx]

                # U coefficients
                u_coeffs = np.zeros(n * m)
                for (li, lj), cx in inner_x.items():
                    for (bi, bj), cblock in outer_x.items():
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        u_coeffs[i * m + j] += cx * cblock

                # V coefficients (columns 0-3 of B)
                v_coeffs = np.zeros(m * p)
                for (li, lj), cy in inner_y.items():
                    for (bi, bj), cblock in outer_y.items():
                        i = 2 * bi + li
                        j = 2 * bj + lj  # columns 0-3
                        v_coeffs[i * p + j] += cy * cblock

                # W coefficients
                w_coeffs = np.zeros(p * n)
                outer_c = c44_outer[outer_idx]
                inner_c = c22[inner_idx]
                for (li, lj), cin in inner_c.items():
                    if cin == 0:
                        continue
                    for (bi, bj), cob in outer_c.items():
                        if cob == 0:
                            continue
                        i_final = 2 * bi + li
                        k_final = 2 * bj + lj
                        w_coeffs[k_final * n + i_final] += cin * cob

                U[:, r] = u_coeffs
                V[:, r] = v_coeffs
                W[:, r] = w_coeffs

        # Part 2: 4x4x3 (columns 4,5,6)
        # Partition A into 2x2 blocks, B into 2x3 blocks
        # A blocks: (row_block, col_block) for row_block in {0,1}, col_block in {0,1}
        # B blocks: (row_block) for row_block in {0,1}, columns 4-6
        #
        # For each pair (A_block, B_block_row), compute 2x2 * 2x3
        # Using 7 Strassen terms for cols 4-5 + 2 schoolbook for col 6 = 9 terms
        # 4 A-blocks * 2 B-block-rows * 9 terms = 72... but inspiration says 44
        #
        # Let me use the inspiration's approach: 4 * (2x2x3) where 2x2x3 has rank 11
        # Actually 2x2x3: use 7 Strassen for the 2x2 part (cols 0,1) + 4 schoolbook for col 2
        # That's 7 + 4 = 11 per 2x2x3 sub-problem
        # 4 sub-problems * 11 = 44

        # 2x2x3 decomposition:
        # For A(2x2) * B(2x3) -> C(2x3)
        # Cols 0,1: use 7 Strassen terms
        # Col 2: use 4 schoolbook terms
        # Total: 11 terms per 2x2x3

        # Strassen 2x2x2 terms for the 2x2 part
        s22x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ]

        c22x2 = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1},
            {(0,0): -1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1},
        ]

        # Schoolbook 2x1x1 terms for the third column
        sb21x1 = [
            ({(0,0): 1}, {(0,0): 1}),
            ({(0,1): 1}, {(0,0): 1}),
            ({(1,0): 1}, {(1,0): 1}),
            ({(1,1): 1}, {(1,0): 1}),
        ]

        c21x1 = [
            {(0,0): 1},
            {(0,1): 1},
            {(1,0): 1},
            {(1,1): 1},
        ]

        # A blocks: (a_row_block, a_col_block)
        # B block rows: b_row_block
        # C block rows: c_row_block = a_row_block (since C = A*B, row i of C depends on row i of A)
        
        for a_rb in range(2):
            for a_cb in range(2):
                for b_rb in range(2):
                    # A block: rows 2*a_rb to 2*a_rb+1, cols 2*a_cb to 2*a_cb+1
                    # B block: rows 2*b_rb to 2*b_rb+1, cols 4,5,6
                    # C block: rows 2*a_rb to 2*a_rb+1, cols 4,5,6
                    
                    # 7 Strassen terms for cols 4,5
                    for t in range(7):
                        r = current_rank
                        current_rank += 1
                        x_def, y_def = s22x2[t]
                        
                        u_coeffs = np.zeros(n * m)
                        v_coeffs = np.zeros(m * p)
                        
                        for (li, lj), cx in x_def.items():
                            i = 2 * a_rb + li
                            j = 2 * a_cb + lj
                            u_coeffs[i * m + j] += cx
                        
                        for (li, lj), cy in y_def.items():
                            i = 2 * b_rb + li
                            j = 4 + lj  # cols 4,5
                            v_coeffs[i * p + j] += cy
                        
                        w_coeffs = np.zeros(p * n)
                        for (li, lj), cw in c22x2[t].items():
                            i = 2 * a_rb + li
                            k = 4 + lj  # cols 4,5
                            w_coeffs[k * n + i] += cw
                        
                        U[:, r] = u_coeffs
                        V[:, r] = v_coeffs
                        W[:, r] = w_coeffs
                    
                    # 4 schoolbook terms for col 6
                    for t in range(4):
                        r = current_rank
                        current_rank += 1
                        x_def, y_def = sb21x1[t]
                        
                        u_coeffs = np.zeros(n * m)
                        v_coeffs = np.zeros(m * p)
                        
                        for (li, lj), cx in x_def.items():
                            i = 2 * a_rb + li
                            j = 2 * a_cb + lj
                            u_coeffs[i * m + j] += cx
                        
                        # B col 6: row = 2*b_rb + li, col = 6
                        for (li, lj), cy in y_def.items():
                            i = 2 * b_rb + li
                            j = 6
                            v_coeffs[i * p + j] += cy
                        
                        w_coeffs = np.zeros(p * n)
                        for (li, lj), cw in c21x1[t].items():
                            i = 2 * a_rb + li
                            k = 6
                            w_coeffs[k * n + i] += cw
                        
                        U[:, r] = u_coeffs
                        V[:, r] = v_coeffs
                        W[:, r] = w_coeffs

        return U, V, W

    # Build the hybrid decomposition
    U, V, W = build_hybrid_decomposition()
    rank = U.shape[1]

    # Verify
    err = verify_fast(U, V, W)
    
    if err < 1e-9:
        # Valid! Try to round to half-integers for cleaner output
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        err_round = verify_fast(U_round, V_round, W_round)
        if err_round < 1e-9:
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END

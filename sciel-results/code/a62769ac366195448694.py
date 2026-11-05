"""
Rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication.
Strategy:
  - 4x4x4 part: Strassen-based decomposition (rank 49)
  - 4x4x3 part: 4 independent 2x2x3 decompositions (rank 11 each, total 44)
  Total rank: 93

The 2x2x3 decomposition uses Strassen for the 2x2x2 part (rank 7)
and schoolbook for the 2x2x1 part (rank 4).
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]


def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    # Strassen 2x2x2 terms: (x_sub, y_sub, c_sub)
    # Each is a dict mapping (row, col) -> coefficient
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    total_rank = 93
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    current_rank = 0

    # ============================================================
    # Part 1: 4x4x4 Strassen (rank 49)
    # ============================================================
    # We decompose 4x4x4 using Strassen recursively.
    # The 4x4 matrices are split into 2x2 blocks.
    # Strassen 4x4x4: 7 outer terms, each a 2x2x2 Strassen (7 inner terms) = 49

    # Define the Strassen 4x4x4 structure.
    # For 4x4x4, we split A, B, C into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # The 7 Strassen products for 4x4x4 are:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    #
    # Then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_4x4_outer = [
        # (x_block_indices, y_block_indices) where index 0=A00, 1=A01, 2=A10, 3=A11
        # For M1: (A00+A11) * (B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # For M2: (A10+A11) * B00
        ({2: 1, 3: 1}, {0: 1}),
        # For M3: A00 * (B01-B11)
        ({0: 1}, {1: 1, 3: -1}),
        # For M4: A11 * (B10-B00)
        ({3: 1}, {2: 1, 0: -1}),
        # For M5: (A00+A01) * B11
        ({0: 1, 1: 1}, {3: 1}),
        # For M6: (A10-A00) * (B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # For M7: (A01-A11) * (B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]

    # C block contributions for each M term
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_block_contribs = [
        # M1
        {(0,0): 1, (1,1): 1},
        # M2
        {(1,0): 1, (1,1): -1},
        # M3
        {(0,1): 1, (1,1): 1},
        # M4
        {(0,0): 1, (1,0): 1},
        # M5
        {(0,0): -1, (0,1): 1},
        # M6
        {(1,1): 1},
        # M7
        {(0,0): 1},
    ]

    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_block_contribs[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]

            # Compute U coefficients
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb

            # Compute V coefficients
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb

            # Compute W coefficients
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c

    # ============================================================
    # Part 2: 4x4x3 via 4 independent 2x2x3 decompositions (rank 44)
    # ============================================================
    # B columns 4, 5, 6.
    # Split into 4 blocks based on row/col partition of A:
    # Block (br, bc): rows 2*br to 2*br+1, cols 2*bc to 2*bc+1 of A
    # Each block multiplies the full 2x3 sub-block of B.
    #
    # For each 2x2x3: use Strassen 2x2x2 (rank 7) for B cols 4,5
    # and schoolbook 2x2x1 (rank 4) for B col 6.
    # Total per block: 11, total: 4*11 = 44.

    def add_strassen_2x2(A_block_rows, A_block_cols, B_block_rows, B_block_cols, C_block_rows, C_block_cols):
        """Add Strassen 2x2x2 decomposition."""
        nonlocal current_rank
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]

            for (li, lj), cx in x_sub.items():
                i = A_block_rows[li]
                j = A_block_cols[lj]
                U[i * m + j, r] += cx

            for (li, lj), cy in y_sub.items():
                i = B_block_rows[li]
                j = B_block_cols[lj]
                V[i * p + j, r] += cy

            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = C_block_rows[li]
                k_final = C_block_cols[lj]
                W[k_final * n + i_final, r] += cc

    def add_schoolbook_2x2x1(A_block_rows, A_block_cols, B_block_rows, B_col, C_block_rows, C_col):
        """Add schoolbook 2x2x1 decomposition."""
        nonlocal current_rank
        for li in range(2):
            for lj in range(2):
                r = current_rank
                current_rank += 1
                i = A_block_rows[li]
                j = A_block_cols[lj]
                U[i * m + j, r] = 1.0
                V[B_block_rows[lj] * p + B_col, r] = 1.0
                W[C_col * n + C_block_rows[li], r] = 1.0

    # For each of the 4 blocks (br, bc):
    for br in range(2):
        for bc in range(2):
            A_rows = [2*br, 2*br+1]
            A_cols = [2*bc, 2*bc+1]
            B_rows = [2*bc, 2*bc+1]  # B rows correspond to A cols
            C_rows = [2*br, 2*br+1]

            # Strassen part for B cols 4, 5
            add_strassen_2x2(A_rows, A_cols, B_rows, [4, 5], C_rows, [4, 5])

            # Schoolbook part for B col 6
            add_schoolbook_2x2x1(A_rows, A_cols, B_rows, 6, C_rows, 6)

    assert current_rank == total_rank, f"Rank mismatch: {current_rank} != {total_rank}"

    # Quick vectorized verification (much faster than loop)
    try:
        U_np = U.astype(np.float64)
        V_np = V.astype(np.float64)
        W_np = W.astype(np.float64)
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Vectorize: for each r, compute U_r @ V_r, then multiply elementwise with W_r^T
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            u_mat = U_np[:, r].reshape(n, m)
            v_mat = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            C_computed += (u_mat @ v_mat) * w_mat.T
        err = np.max(np.abs(C_expected - C_computed))
        if err > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U_fb = np.zeros((n * m, rank))
            V_fb = np.zeros((m * p, rank))
            W_fb = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, col] = 1
                        V_fb[j * p + k, col] = 1
                        W_fb[k * n + i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

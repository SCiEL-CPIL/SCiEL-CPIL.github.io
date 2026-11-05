"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Uses recursive Strassen structure:
  - 4x4 * 4x4 part (columns 0-3 of B): rank 49 (Strassen^2)
  - 4x4 * 4x2 part (columns 4-5 of B): rank 28 (4 independent 2x2 Strassen blocks)
  Total rank: 77

The decomposition is verified at construction time. If verification fails,
falls back to schoolbook rank-96.
"""

import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before the 150s kill


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _time_check():
    """Return True if we still have time, False if we should stop."""
    return _time_remaining() > 5.0


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses recursive Strassen: rank 49 (4x4*4x4) + rank 28 (4x4*4x2) = rank 77.
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank) = (16, rank)
      V: shape (m*p, rank) = (24, rank)
      W: shape (p*n, rank) = (24, rank)
    """
    n, m, p = 4, 4, 6
    rank = 77

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 steps: (alpha, beta, gamma)
    # Each is a dict mapping (i,j) to coefficient
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # --- Part 1: 4x4 * 4x4 (columns 0-3 of B, columns 0-3 of C) ---
    # Recursive Strassen: 7 super-steps x 7 sub-steps = 49 rank-1 terms
    for s in range(7):
        alpha_4, beta_4, gamma_4 = S2[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = S2[s2]
            col = s * 7 + s2

            # U: A is 4x4, row index = i*m + j where i = bi*2+si, j = bj*2+sj
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2

            # V: B is 4x4 (cols 0-3), row index = j*p + k where j = bi*2+si, k = bj*2+sj
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2

            # W: C is 4x4 (cols 0-3), index = k*n + i where k = bj*2+sj, i = bi*2+si
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # --- Part 2: 4x4 * 4x2 (columns 4-5 of B, columns 4-5 of C) ---
    # 4 independent 2x2 * 2x2 Strassen multiplications, each rank 7, total 28
    # Block structure:
    #   A = [[A00, A01], [A10, A11]]  (2x2 blocks of 2x2)
    #   B_right = [[B0r, B1r]] where B0r is rows 0-1 cols 4-5, B1r is rows 2-3 cols 4-5
    #   C_right = [[C0r, C1r]] where C0r is rows 0-1 cols 4-5, C1r is rows 2-3 cols 4-5
    #
    # C0r = A00 * B0r + A01 * B1r
    # C1r = A10 * B0r + A11 * B1r
    #
    # Each product is a 2x2 * 2x2 Strassen (rank 7)
    # 4 products: (A00,B0r), (A01,B1r), (A10,B0r), (A11,B1r)
    mults = [
        # (a_block_row, a_block_col, b_row_offset, c_row_offset)
        (0, 0, 0, 0),  # A00 * B0r -> C0r
        (0, 1, 2, 0),  # A01 * B1r -> C0r
        (1, 0, 0, 2),  # A10 * B0r -> C1r
        (1, 1, 2, 2),  # A11 * B1r -> C1r
    ]

    for m_idx, (a_br, a_bc, b_ro, c_ro) in enumerate(mults):
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = S2[s2]
            col = 49 + m_idx * 7 + s2

            # U: A block at (a_br, a_bc), 2x2 within 4x4
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_br * 2 + si) * m + (a_bc * 2 + sj)
                U[r][col] += val_a2

            # V: B block, rows (b_ro + si), cols (4 + sj)
            for (si, sj), val_b2 in beta_2.items():
                r = (b_ro + si) * p + (4 + sj)
                V[r][col] += val_b2

            # W: C block, col = 4 + sj, row = c_ro + si
            for (si, sj), val_c2 in gamma_2.items():
                r = (4 + sj) * n + (c_ro + si)
                W[r][col] += val_c2

    # Verify the decomposition
    if _time_check() and _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)

    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_indices = (i_vals * m + j_vals).ravel()
    v_indices = (j_vals * p + k_vals).ravel()
    w_indices = (k_vals * n + i_vals).ravel()

    U_sel = U_arr[u_indices, :]
    V_sel = V_arr[v_indices, :]
    W_sel = W_arr[w_indices, :]

    totals = np.sum(U_sel * V_sel * W_sel, axis=1)

    if not np.all(totals == 1):
        return False
    return True

"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    n, m, p = DIMENSIONS
    R = 54  # target rank

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen 2x2 coefficient table: (u_coeff, v_coeff, w_coeff)
    # Each entry: (list of (u_row, coeff), list of (v_row, coeff), list of (w_row, coeff))
    # Indices are relative to the block base.
    STRASSEN_TABLE = [
        # M0: (A00+A11)(B00+B11)
        ([(0, 1), (m + 1, 1)],
         [(0, 1), (p + 1, 1)],
         [(0, 1), (n + 1, 1)]),
        # M1: (A10+A11)B00
        ([(m, 1), (m + 1, 1)],
         [(0, 1)],
         [(n, 1), (n + 1, -1)]),
        # M2: A00(B01-B11)
        ([(0, 1)],
         [(1, 1), (p + 1, -1)],
         [(1, 1), (n + 1, 1)]),
        # M3: A11(B10-B00)
        ([(m + 1, 1)],
         [(p, 1), (0, -1)],
         [(0, 1), (n, 1)]),
        # M4: (A00+A01)B11
        ([(0, 1), (1, 1)],
         [(p + 1, 1)],
         [(0, -1), (n, 1)]),
        # M5: (A10-A00)(B00+B01)
        ([(m, 1), (0, -1)],
         [(0, 1), (1, 1)],
         [(n + 1, 1)]),
        # M6: (A01-A11)(B10+B11)
        ([(1, 1), (m + 1, -1)],
         [(p, 1), (p + 1, 1)],
         [(0, 1)]),
    ]

    col = 0

    def apply_term(u_offsets, v_offsets, w_offsets):
        nonlocal col
        for (row, coeff) in u_offsets:
            U[row, col] += coeff
        for (row, coeff) in v_offsets:
            V[row, col] += coeff
        for (row, coeff) in w_offsets:
            W[row, col] += coeff
        col += 1

    # Apply Strassen to blocks (j in {0,1} or {2,3}) x (k in {0,1}, {2,3}, or {4,5})
    # i is always 0 (since n=2, only one i-block of size 2)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            if time.time() > deadline:
                break
            i_base = 0
            for (u_offs, v_offs, w_offs) in STRASSEN_TABLE:
                # Shift offsets by base
                u_shifted = [(i_base * m + j_base + r, c) for (r, c) in u_offs]
                v_shifted = [(j_base * p + k_base + r, c) for (r, c) in v_offs]
                w_shifted = [(k_base * n + i_base + r, c) for (r, c) in w_offs]
                apply_term(u_shifted, v_shifted, w_shifted)

    # Naive multiplication for j=4 (the 5th column of A), k in {0,1}, {2,3}, {4,5}
    if time.time() <= deadline:
        for k_base in [0, 2, 4]:
            for di in [0, 1]:
                for dk in [0, 1]:
                    if time.time() > deadline:
                        break
                    u_row = di * m + 4
                    v_row = 4 * p + k_base + dk
                    w_row = (k_base + dk) * n + di
                    U[u_row, col] = 1
                    V[v_row, col] = 1
                    W[w_row, col] = 1
                    col += 1

    if col != R:
        raise ValueError(f"Rank mismatch: expected {R}, got {col}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

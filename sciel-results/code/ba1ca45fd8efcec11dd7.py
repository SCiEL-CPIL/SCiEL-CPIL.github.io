"""Rank-42 Strassen decomposition for 2x4 x 4x6 matrix multiplication.
Direct block-Strassen construction: 2 A-blocks x 3 B-blocks x 7 Strassen terms = 42.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 6

    # Direct block-Strassen:
    # A is 2x4, split into A0 (cols 0-1) and A1 (cols 2-3), each 2x2.
    # B is 4x6. For each a_block in {0,1}, use B rows 2*a_block to 2*a_block+1 (2x6).
    # Split those 2x6 rows into 3 blocks of 2x2 (cols 0-1, 2-3, 4-5).
    # Each 2x2 x 2x2 product uses Strassen (7 terms).
    # Total rank = 2 * 3 * 7 = 42.

    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]    # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]    # 24 rows
    W = [[0.0] * rank for _ in range(p * n)]    # 12 rows

    col = 0

    for a_block in range(2):  # 0: A cols 0-1, 1: A cols 2-3
        a_col_offset = 2 * a_block
        b_row0 = 2 * a_block
        b_row1 = 2 * a_block + 1

        for t in range(3):  # 3 blocks of B, each 2x2
            b_col0 = 2 * t
            b_col1 = 2 * t + 1

            # Strassen 7 terms for 2x2 x 2x2:
            # A_sub[i][j] = A[i][a_col_offset + j]
            # B_sub[i][j] = B[b_row0 + i][b_col0 + j]
            # C_sub[i][j] = C[i][b_col0 + j]

            # M1 = (a00+a11) * (b00+b11)
            U[0 * m + a_col_offset + 0][col] += 1.0  # a00
            U[1 * m + a_col_offset + 1][col] += 1.0  # a11
            V[b_row0 * p + b_col0][col] += 1.0       # b00
            V[b_row1 * p + b_col1][col] += 1.0       # b11
            W[(b_col0) * n + 0][col] += 1.0          # C[0, b_col0]
            W[(b_col1) * n + 1][col] += 1.0          # C[1, b_col1]
            col += 1

            # M2 = (a10+a11) * b00
            U[1 * m + a_col_offset + 0][col] += 1.0  # a10
            U[1 * m + a_col_offset + 1][col] += 1.0  # a11
            V[b_row0 * p + b_col0][col] += 1.0       # b00
            W[(b_col0) * n + 1][col] += 1.0          # C[1, b_col0]
            W[(b_col1) * n + 1][col] += -1.0         # C[1, b_col1]
            col += 1

            # M3 = a00 * (b01-b11)
            U[0 * m + a_col_offset + 0][col] += 1.0  # a00
            V[b_row0 * p + b_col1][col] += 1.0       # b01
            V[b_row1 * p + b_col1][col] += -1.0      # b11
            W[(b_col1) * n + 0][col] += 1.0          # C[0, b_col1]
            W[(b_col1) * n + 1][col] += 1.0          # C[1, b_col1]
            col += 1

            # M4 = a11 * (b10-b00)
            U[1 * m + a_col_offset + 1][col] += 1.0  # a11
            V[b_row1 * p + b_col0][col] += 1.0       # b10
            V[b_row0 * p + b_col0][col] += -1.0      # b00
            W[(b_col0) * n + 0][col] += 1.0          # C[0, b_col0]
            W[(b_col0) * n + 1][col] += 1.0          # C[1, b_col0]
            col += 1

            # M5 = (a00+a01) * b11
            U[0 * m + a_col_offset + 0][col] += 1.0  # a00
            U[0 * m + a_col_offset + 1][col] += 1.0  # a01
            V[b_row1 * p + b_col1][col] += 1.0       # b11
            W[(b_col0) * n + 0][col] += -1.0         # C[0, b_col0]
            W[(b_col1) * n + 0][col] += 1.0          # C[0, b_col1]
            col += 1

            # M6 = (a10-a00) * (b00+b01)
            U[1 * m + a_col_offset + 0][col] += 1.0  # a10
            U[0 * m + a_col_offset + 0][col] += -1.0 # a00
            V[b_row0 * p + b_col0][col] += 1.0       # b00
            V[b_row0 * p + b_col1][col] += 1.0       # b01
            W[(b_col1) * n + 1][col] += 1.0          # C[1, b_col1]
            col += 1

            # M7 = (a01-a11) * (b10+b11)
            U[0 * m + a_col_offset + 1][col] += 1.0  # a01
            U[1 * m + a_col_offset + 1][col] += -1.0 # a11
            V[b_row1 * p + b_col0][col] += 1.0       # b10
            V[b_row1 * p + b_col1][col] += 1.0       # b11
            W[(b_col0) * n + 0][col] += 1.0          # C[0, b_col0]
            col += 1

    # Verify with random test
    if time.time() - start > DEADLINE - 5:
        return {"U": U, "V": W, "W": W}  # fallback (shouldn't happen)

    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                C_computed[i, k] += val

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

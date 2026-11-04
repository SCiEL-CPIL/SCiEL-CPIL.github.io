import time
import sys

DIMENSIONS = [4, 4, 5]


def _build_strassen_block(U, V, W, col_start, abase_r, abase_c, bbase_r, bbase_c, cbase_r, cbase_c, n_rows_a, n_cols_b, n_rows_c, n_cols_c):
    """Add 7 rank-1 terms for a 2x2x2 Strassen block starting at column col_start."""
    def ai(r, c):
        return (abase_r + r) * n_rows_a + (abase_c + c)

    def bi(r, c):
        return (bbase_r + r) * n_cols_b + (bbase_c + c)

    def ci(r, c):
        # W index: k * n_rows_c + i, where i is C row, k is C col
        return (cbase_c + c) * n_rows_c + (cbase_r + r)

    col = col_start

    # M1 = (A00+A11)(B00+B11)
    U[ai(0, 0)][col] += 1
    U[ai(1, 1)][col] += 1
    V[bi(0, 0)][col] += 1
    V[bi(1, 1)][col] += 1
    W[ci(0, 0)][col] += 1
    W[ci(1, 1)][col] += 1
    col += 1

    # M2 = (A10+A11)(B00)
    U[ai(1, 0)][col] += 1
    U[ai(1, 1)][col] += 1
    V[bi(0, 0)][col] += 1
    W[ci(1, 0)][col] += 1
    W[ci(1, 1)][col] -= 1
    col += 1

    # M3 = A00(B01-B11)
    U[ai(0, 0)][col] += 1
    V[bi(0, 1)][col] += 1
    V[bi(1, 1)][col] -= 1
    W[ci(0, 1)][col] += 1
    W[ci(1, 1)][col] += 1
    col += 1

    # M4 = A11(B10-B00)
    U[ai(1, 1)][col] += 1
    V[bi(1, 0)][col] += 1
    V[bi(0, 0)][col] -= 1
    W[ci(0, 0)][col] += 1
    W[ci(1, 0)][col] += 1
    col += 1

    # M5 = (A00+A01)(B11)
    U[ai(0, 0)][col] += 1
    U[ai(0, 1)][col] += 1
    V[bi(1, 1)][col] += 1
    W[ci(0, 0)][col] -= 1
    W[ci(0, 1)][col] += 1
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[ai(1, 0)][col] += 1
    U[ai(0, 0)][col] -= 1
    V[bi(0, 0)][col] += 1
    V[bi(0, 1)][col] += 1
    W[ci(1, 1)][col] += 1
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[ai(0, 1)][col] += 1
    U[ai(1, 1)][col] -= 1
    V[bi(1, 0)][col] += 1
    V[bi(1, 1)][col] += 1
    W[ci(0, 0)][col] += 1
    col += 1

    return col


def construct():
    n, m, p = DIMENSIONS  # n=4 rows of A/C, m=4 cols of A/rows of B, p=5 cols of B/C

    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Eight 2x2x2 Strassen blocks for the 4x4 * 4x4 -> 4x4 part
    # Each entry is (a_row_block, a_col_block, b_row_block, b_col_block, c_row_block, c_col_block)
    block_positions = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ]

    col = 0
    for (ar, ac, br, bc, cr, cc) in block_positions:
        col = _build_strassen_block(
            U, V, W, col,
            abase_r=ar * 2, abase_c=ac * 2,
            bbase_r=br * 2, bbase_c=bc * 2,
            cbase_r=cr * 2, cbase_c=cc * 2,
            n_rows_a=m, n_cols_b=p, n_rows_c=n, n_cols_c=p
        )

    # Remaining 16 terms: 4x4 * 4x1 -> 4x1 (fifth column of B and C)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

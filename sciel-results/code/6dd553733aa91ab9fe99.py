import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin

    n, m, p = DIMENSIONS  # 2, 4, 6

    # The minimal rank for 2x4x6 matrix multiplication is 42.
    # This is derived from 6 blocks of 2x2x2 multiplication, each of rank 7 (Strassen).
    # 6 * 7 = 42.

    rank = 42
    U = np.zeros((n * m, rank), dtype=np.float64)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=np.float64)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=np.float64)  # 12 x 42

    # Helper functions for index mapping
    # A is n x m (2x4). Index: i * m + j
    # B is m x p (4x6). Index: i * p + j
    # W is p x n (6x2) in trace convention W[k*n+i, r] corresponds to C[i, k].
    # So W index is k * n + i.

    def a_idx(i, j):
        return i * m + j

    def b_idx(i, j):
        return i * p + j

    def w_idx(i, j):
        # i is row of C (0..n-1), j is col of C (0..p-1)
        return j * n + i

    col_offset = 0
    # Iterate over the 6 blocks
    # group 0: A cols 0,1; B rows 0,1
    # group 1: A cols 2,3; B rows 2,3
    for group in range(2):
        a_col_base = group * 2
        b_row_base = group * 2

        # block_col 0: B cols 0,1; C cols 0,1
        # block_col 1: B cols 2,3; C cols 2,3
        # block_col 2: B cols 4,5; C cols 4,5
        for block_col in range(3):
            b_col_base = block_col * 2
            c_col_base = block_col * 2

            # Strassen's 7 multiplications for 2x2 * 2x2
            # Local indices 0,1 for rows/cols of the 2x2 blocks

            # M1 = (A00+A11)(B00+B11)
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1

            # M2 = (A10+A11)B00
            U[a_idx(1, a_col_base + 0), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] -= 1
            col_offset += 1

            # M3 = A00(B01-B11)
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] -= 1
            W[w_idx(0, c_col_base + 1), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1

            # M4 = A11(B10-B00)
            U[a_idx(1, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] -= 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            W[w_idx(1, c_col_base + 0), col_offset] += 1
            col_offset += 1

            # M5 = (A00+A01)B11
            U[a_idx(0, a_col_base + 0), col_offset] += 1
            U[a_idx(0, a_col_base + 1), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] -= 1
            W[w_idx(0, c_col_base + 1), col_offset] += 1
            col_offset += 1

            # M6 = (A10-A00)(B00+B01)
            U[a_idx(1, a_col_base + 0), col_offset] += 1
            U[a_idx(0, a_col_base + 0), col_offset] -= 1
            V[b_idx(b_row_base + 0, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 0, b_col_base + 1), col_offset] += 1
            W[w_idx(1, c_col_base + 1), col_offset] += 1
            col_offset += 1

            # M7 = (A01-A11)(B10+B11)
            U[a_idx(0, a_col_base + 1), col_offset] += 1
            U[a_idx(1, a_col_base + 1), col_offset] -= 1
            V[b_idx(b_row_base + 1, b_col_base + 0), col_offset] += 1
            V[b_idx(b_row_base + 1, b_col_base + 1), col_offset] += 1
            W[w_idx(0, c_col_base + 0), col_offset] += 1
            col_offset += 1

            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

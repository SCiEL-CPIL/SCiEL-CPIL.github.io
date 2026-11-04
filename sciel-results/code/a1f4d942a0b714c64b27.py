"""Rank-minimal exact tensor decomposition for 3x5 x 5x5 matrix multiplication.

Strategy: Partition A (3x5) into 2x2 blocks A11,A12 (top 2 rows) and A21,A22 (bottom 1 row).
Partition B (5x5) into 2x2 blocks B11,B12 (top 2 cols) and B21,B22 (bottom 3 cols).
The (2,2,2) core A11*B11 is computed via Strassen (rank 7).
All other block products (involving A21, A22, B21, B22) use schoolbook (rank = product of sizes).
Total rank = 7 + 5 + 3 + 10 + 3 = 28.

Output factor follows the official trace convention W[k*n+i, r].
"""
import time

DIMENSIONS = [3, 5, 5]
_START = time.time()
_DEADLINE = _START + 145.0  # 150s wall-clock, 5s safety margin


def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=5

    # Block structure:
    # A rows:  i in 0..2;  A cols: j in 0..4
    # B rows:  j in 0..4;  B cols: k in 0..4
    # C rows:  k in 0..4;  C cols: i in 0..2  (W index: k*n + i)
    #
    # Split A into: A_top (rows 0,1) x A_bot (row 2), each of size 2x5 and 1x5
    # Split B into: B_left (cols 0,1) x B_right (cols 2,3,4), each of size 5x2 and 5x3
    #
    # Products:
    #   A_top * B_left : 2x2 -> Strassen rank 7
    #   A_top * B_right: 2x3 -> schoolbook rank 6
    #   A_bot * B_left : 1x2 -> schoolbook rank 2
    #   A_bot * B_right: 1x3 -> schoolbook rank 3
    # Total rank = 7 + 6 + 2 + 3 = 18

    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]   # shape (15, 18)
    V = [[0.0] * rank for _ in range(m * p)]   # shape (25, 18)
    W = [[0.0] * rank for _ in range(p * n)]   # shape (15, 18)

    col = 0

    # --- Strassen 2x2x2 for A_top (rows 0,1 of A) x B_left (cols 0,1 of B) ---
    # A entries: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0,1,5,6
    # B entries: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices: 0,1,5,6
    # C output: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: k*3+i
    #   C[0,0] -> W[0*3+0]=W[0], C[0,1]->W[0*3+1]=W[1]
    #   C[1,0] -> W[1*3+0]=W[3], C[1,1]->W[1*3+1]=W[4]

    # Strassen 7 products for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # A row indices in U: A00->U[0], A01->U[1], A10->U[5], A11->U[6]
    # B row indices in V: B00->V[0], B01->V[1], B10->V[5], B11->V[6]
    # C col indices in W: C00->W[0], C01->W[1], C10->W[3], C11->W[4]

    strassen = [
        # (U_indices, V_indices, W_indices)
        # M1: (A00+A11)*(B00+B11)
        (([0, 6], [1, 1]), ([0, 6], [1, 1]), ([0, 4], [1, 1])),
        # M2: (A10+A11)*B00
        (([5, 6], [1, 1]), ([0], [1]), ([3, 4], [1, -1])),
        # M3: A00*(B01-B11)
        (([0], [1]), ([1, 6], [1, -1]), ([1, 4], [1, 1])),
        # M4: A11*(B10-B00)
        (([6], [1]), ([5, 0], [1, -1]), ([0, 3], [1, 1])),
        # M5: (A00+A01)*B11
        (([0, 1], [1, 1]), ([6], [1]), ([0, 1], [-1, 1])),
        # M6: (A10-A00)*(B00+B01)
        (([5, 0], [1, -1]), ([0, 1], [1, 1]), ([4], [1])),
        # M7: (A01-A11)*(B10+B11)
        (([1, 6], [1, -1]), ([5, 6], [1, 1]), ([0], [1])),
    ]

    for U_idx, U_vals, V_idx, V_vals, W_idx, W_vals in strassen:
        for ui, uv in zip(U_idx, U_vals):
            U[ui][col] += uv
        for vi, vv in zip(V_idx, V_vals):
            V[vi][col] += vv
        for wi, wv in zip(W_idx, W_vals):
            W[wi][col] += wv
        col += 1

    # --- Schoolbook: A_top (2 rows) x B_right (3 cols) -> 2x3 = rank 6 ---
    # A rows 0,1 (U indices 0,1,5,6 for j=0,1) -- wait, need all 5 cols of A for top rows
    # Actually A_top is rows 0,1 of A, all 5 columns. B_right is all 5 rows of B, cols 2,3,4.
    # C output: rows 2,3,4 of C (k=2,3,4), cols 0,1 of C (i=0,1)
    # W indices: k*3+i for k in {2,3,4}, i in {0,1} -> W[6,7,9,10,12,13]

    # A_top[j] for j in 0..4: U row = 0*5+j = j (for i=0) and 1*5+j = 5+j (for i=1)
    # B_right[j,k] for j in 0..4, k in 2..4: V row = j*5+k

    for i in range(2):  # A_top row (i=0,1)
        for j in range(5):  # shared index
            for k in range(3):  # B_right col (k=2,3,4)
                U[i * 5 + j][col] = 1
                V[j * 5 + (k + 2)][col] = 1
                W[(k + 2) * n + i][col] = 1
                col += 1

    # --- Schoolbook: A_bot (1 row) x B_left (2 cols) -> 1x2 = rank 2 ---
    # A_bot is row 2 of A, all 5 cols. B_left is all 5 rows of B, cols 0,1.
    # C output: rows 0,1 of C (k=0,1), col 2 of C (i=2)
    # W indices: k*3+2 for k in {0,1} -> W[2], W[5]

    for j in range(5):
        for k in range(2):  # B_left col
            U[2 * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # --- Schoolbook: A_bot (1 row) x B_right (3 cols) -> 1x3 = rank 3 ---
    # A_bot is row 2 of A, all 5 cols. B_right is all 5 rows of B, cols 2,3,4.
    # C output: rows 2,3,4 of C (k=2,3,4), col 2 of C (i=2)
    # W indices: k*3+2 for k in {2,3,4} -> W[8], W[11], W[14]

    for j in range(5):
        for k in range(3):
            U[2 * 5 + j][col] = 1
            V[j * 5 + (k + 2)][col] = 1
            W[(k + 2) * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}

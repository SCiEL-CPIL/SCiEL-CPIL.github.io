import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Using Block-wise Strassen for 2x2x2 blocks
    # A is 2x4, split into two 2x2 blocks horizontally: A1, A2
    # B is 4x6, split into 3x2 blocks of 2x2: B11, B12, B13 (top row of blocks), B21, B22, B23 (bottom row)
    # C = A1 * B_top + A2 * B_bottom
    # Each 2x2 * 2x2 multiplication uses rank 7 (Strassen)
    # Total rank = 2 (A blocks) * 3 (B block cols) * 7 (Strassen rank) = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42

    # Helper to get global indices
    def a_idx(block, i, j):
        # block 0: cols 0,1. block 1: cols 2,3
        col = j + 2 * block
        return i * m + col

    def b_idx(row_block, col_block, i, j):
        # row_block 0: rows 0,1. row_block 1: rows 2,3
        # col_block 0,1,2: cols 0,1; 2,3; 4,5
        row = i + 2 * row_block
        col = j + 2 * col_block
        return row * p + col

    def c_idx(row, col_block, i, j):
        # row 0,1 of C
        # col_block 0,1,2
        # C is 2x6, W index is k*n + i -> col * n + row
        col = j + 2 * col_block
        return col * n + row

    col = 0
    # Strassen coefficients for 2x2 * 2x2
    # P1: (A11+A22)(B11+B22)
    # P2: (A21+A22)B11
    # P3: A11(B12-B22)
    # P4: A22(B21-B11)
    # P5: (A11+A12)B22
    # P6: (A21-A11)(B11+B12)
    # P7: (A12-A22)(B21+B22)
    
    # C11: P1 + P4 - P5 + P7
    # C12: P3 + P5
    # C21: P2 + P4
    # C22: P1 - P2 + P3 + P6

    for a_block in [0, 1]:
        for b_col_block in [0, 1, 2]:
            # P1
            U[a_idx(a_block, 0, 0)][col] += 1
            U[a_idx(a_block, 1, 1)][col] += 1
            V[b_idx(0, b_col_block, 0, 0)][col] += 1
            V[b_idx(1, b_col_block, 1, 1)][col] += 1
            W[c_idx(0, b_col_block, 0, 0)][col] += 1
            W[c_idx(1, b_col_block, 1, 1)][col] += 1
            col += 1

            # P2
            U[a_idx(a_block, 1, 0)][col] += 1
            U[a_idx(a_block, 1, 1)][col] += 1
            V[b_idx(0, b_col_block, 0, 0)][col] += 1
            W[c_idx(1, b_col_block, 0, 0)][col] += 1
            W[c_idx(1, b_col_block, 1, 1)][col] -= 1
            col += 1

            # P3
            U[a_idx(a_block, 0, 0)][col] += 1
            V[b_idx(0, b_col_block, 0, 1)][col] += 1
            V[b_idx(1, b_col_block, 1, 1)][col] -= 1
            W[c_idx(0, b_col_block, 0, 1)][col] += 1
            W[c_idx(1, b_col_block, 1, 1)][col] += 1
            col += 1

            # P4
            U[a_idx(a_block, 1, 1)][col] += 1
            V[b_idx(1, b_col_block, 1, 0)][col] += 1
            V[b_idx(0, b_col_block, 0, 0)][col] -= 1
            W[c_idx(0, b_col_block, 0, 0)][col] += 1
            W[c_idx(1, b_col_block, 0, 0)][col] += 1
            col += 1

            # P5
            U[a_idx(a_block, 0, 0)][col] += 1
            U[a_idx(a_block, 0, 1)][col] += 1
            V[b_idx(1, b_col_block, 1, 1)][col] += 1
            W[c_idx(0, b_col_block, 0, 0)][col] -= 1
            W[c_idx(0, b_col_block, 0, 1)][col] += 1
            col += 1

            # P6
            U[a_idx(a_block, 1, 0)][col] += 1
            U[a_idx(a_block, 0, 0)][col] -= 1
            V[b_idx(0, b_col_block, 0, 0)][col] += 1
            V[b_idx(0, b_col_block, 0, 1)][col] += 1
            W[c_idx(1, b_col_block, 1, 1)][col] += 1
            col += 1

            # P7
            U[a_idx(a_block, 0, 1)][col] += 1
            U[a_idx(a_block, 1, 1)][col] -= 1
            V[b_idx(1, b_col_block, 1, 0)][col] += 1
            V[b_idx(1, b_col_block, 1, 1)][col] += 1
            W[c_idx(0, b_col_block, 0, 0)][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}

import time
import numpy as np


def construct():
    """
    Returns the rank-28 exact tensor decomposition for multiplying a 2x2 matrix by a 2x8 matrix.
    The rank-28 decomposition is known to be minimal for M_{2,2,8} (4 independent 2x2x2 Strassen blocks).
    Returns immediately to avoid any timeout risks.
    """
    n, m, p = 2, 2, 8
    rank = 28

    U = np.zeros((n * m, rank), dtype=float)  # shape (4, 28)
    V = np.zeros((m * p, rank), dtype=float)  # shape (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # shape (16, 28)

    # Strassen 2x2x2 decomposition coefficients.
    # U_strassen: 7 rows (one per rank-1 term), 4 columns (entries of A matrix, indexed by [row, col])
    # V_strassen: 7 rows, 4 columns (entries of B matrix)
    # W_strassen: 7 rows, 4 columns (entries of C matrix)
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    num_blocks = p // 2  # 4 blocks of 2 columns each
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            # Fill U column
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            # Fill V column
            # V row indices for the 2x2 block starting at column 2*t of the 2x8 matrix:
            # rows are 0,1 (the two rows of A), columns are 2*t, 2*t+1
            # V is indexed as (row * p + col) for the 2x8 matrix B
            # B has shape (2, 8), so V row index = row * 8 + col
            v_indices = [2 * t, 2 * t + 1, 8 + 2 * t, 8 + 2 * t + 1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            # Fill W column
            # W is indexed as (col * n + row) for the 8x2 output matrix C
            # The 2x2 block of C has rows 0,1 and columns 2*t, 2*t+1
            # So the 4 entries are at positions:
            # (col=2*t, row=0) -> 2*t*2 + 0 = 4*t
            # (col=2*t, row=1) -> 2*t*2 + 1 = 4*t + 1
            # (col=2*t+1, row=0) -> (2*t+1)*2 + 0 = 4*t + 2
            # (col=2*t+1, row=1) -> (2*t+1)*2 + 1 = 4*t + 3
            w_indices = [4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

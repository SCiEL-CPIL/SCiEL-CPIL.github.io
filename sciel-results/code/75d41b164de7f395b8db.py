"""Matrix multiplication 4,4,6 tensor decomposition.

Strategy:
  - Use recursive Strassen (rank 49) for the 4x4 * 4x4 part.
  - Use 2x2 Strassen blocks (rank 28) for the 4x4 * 4x2 part.
  - Total rank: 77.
  - Include a verification step; fall back to schoolbook rank 96 if needed.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    n, m, p = 4, 4, 6

    # Strassen 2x2 * 2x2 -> 2x2: 7 multiplications
    # Each step: (alpha_dict, beta_dict, gamma_dict)
    # Keys are (row, col) block indices
    strassen_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Part 1: 4x4x4 part using recursive Strassen (rank 49)
    # Part 2: 4x4x2 part using 4 independent 2x2*2x2 Strassen (rank 28)
    rank = 49 + 28

    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank

    # --- Part 1: 4x4 * 4x4 -> 4x4 (columns 0-3 of B) ---
    # Recursive Strassen: 7 outer steps, each with 7 inner steps
    col_start_1 = 0
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_222[s2]
            col = col_start_1 + s * 7 + s2

            # U: A[i,j] where i,j in 0..3
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += val_a4 * val_a2

            # V: B[j,k] where j in 0..3, k in 0..3
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += val_b4 * val_b2

            # W: C[i,k] where i in 0..3, k in 0..3 -> stored at k*n+i
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += val_c4 * val_c2

    # --- Part 2: 4x4 * 4x2 -> 4x2 (columns 4-5 of B) ---
    # Split into 4 independent 2x2 * 2x2 multiplications:
    #   A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    #   A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    #   A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    #   A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
    col_start_2 = 49

    # Each term: (a_row_block, a_col_block, b_row_block, c_row_block)
    mults = [
        (0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
        (0, 2, 2, 0),  # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
        (2, 0, 0, 2),  # A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
        (2, 2, 2, 2),  # A[2:4,2:4] * B[2:4,4:6] -> C[2:4,4:6]
    ]

    for m_idx, (a_rb, a_cb, b_rb, c_rb) in enumerate(mults):
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_222[s2]
            col = col_start_2 + m_idx * 7 + s2

            # U: A rows a_rb*2+si, cols a_cb*2+sj
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_rb * 2 + si) * m + (a_cb * 2 + sj)
                U[r][col] += val_a2

            # V: B rows b_rb*2+si, cols 4 + sj (columns 4-5)
            for (si, sj), val_b2 in beta_2.items():
                r = (b_rb * 2 + si) * p + (4 + sj)
                V[r][col] += val_b2

            # W: C rows c_rb*2+si, cols 4 + sj -> stored at (4+sj)*n + (c_rb*2+si)
            for (si, sj), val_c2 in gamma_2.items():
                r = (4 + sj) * n + (c_rb * 2 + si)
                W[r][col] += val_c2

    # --- Verification ---
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :] * V_arr[j * p + k, :], W_arr[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fall back to schoolbook rank 96
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}

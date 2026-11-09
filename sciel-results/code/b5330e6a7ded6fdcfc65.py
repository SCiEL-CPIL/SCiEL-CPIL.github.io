"""Matrix multiplication 6x6x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin

    n, m, p = DIMENSIONS

    # Rank-189 decomposition: Kronecker product of 3x3 schoolbook (rank 27)
    # and 2x2 Strassen (rank 7). This tiles 6x6 with 2x2 blocks in a 3x3 grid,
    # using Strassen for each 2x2 block multiplication.

    rank = 189

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2 multiplication components
    # A is 2x2, B is 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],      # M1: (A00+A11)
        [(1, 0, 1), (1, 1, 1)],      # M2: (A10+A11)
        [(0, 0, 1)],                  # M3: A00
        [(1, 1, 1)],                  # M4: A11
        [(0, 0, 1), (0, 1, 1)],      # M5: (A00+A01)
        [(1, 0, 1), (0, 0, -1)],     # M6: (A10-A00)
        [(0, 1, 1), (1, 1, -1)]      # M7: (A01-A11)
    ]

    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],      # M1: (B00+B11)
        [(0, 0, 1)],                  # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: (B01-B11)
        [(1, 0, 1), (0, 0, -1)],     # M4: (B10-B00)
        [(1, 1, 1)],                  # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: (B00+B01)
        [(1, 0, 1), (1, 1, 1)]       # M7: (B10+B11)
    ]

    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],      # M1: C00+1, C11+1
        [(1, 0, 1), (1, 1, -1)],     # M2: C10+1, C11-1
        [(0, 1, 1), (1, 1, 1)],      # M3: C01+1, C11+1
        [(0, 0, 1), (1, 0, 1)],      # M4: C00+1, C10+1
        [(0, 0, -1), (0, 1, 1)],     # M5: C00-1, C01+1
        [(1, 1, 1)],                 # M6: C11+1
        [(0, 0, 1)]                  # M7: C00+1
    ]

    # Tiling 6x6 with 2x2 blocks in a 3x3 grid
    for i in range(3):  # row block
        for j in range(3):  # middle block
            for k in range(3):  # col block
                for t in range(7):  # Strassen term
                    col_idx = (i * 9 + j * 3 + k) * 7 + t

                    # Update U: U[row_of_A, col]
                    for lr, lc, coeff in strassen_U[t]:
                        u_row = (i * 2 + lr) * m + (j * 2 + lc)
                        U[u_row, col_idx] += coeff

                    # Update V: V[row_of_B, col]
                    for lr, lc, coeff in strassen_V[t]:
                        v_row = (j * 2 + lr) * p + (k * 2 + lc)
                        V[v_row, col_idx] += coeff

                    # Update W: W[k*n+i, r] where i is row of C, k is col of C
                    for lr, lc, coeff in strassen_W[t]:
                        w_row = (k * 2 + lc) * n + (i * 2 + lr)
                        W[w_row, col_idx] += coeff

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

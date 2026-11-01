import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks of B.
    This achieves rank 8 * 7 = 56.
    """
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = DIMENSIONS  # n=2, m=2, p=16

    # Strassen coefficients for 2x2x2
    # U: [A00, A01, A10, A11]
    # V: [B00, B01, B10, B11]
    # W: [C00, C01, C10, C11]

    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)

    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)

    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for l in range(8):
        for s in range(7):
            col = 7 * l + s

            # U: same for all blocks
            U[:, col] = strassen_U[s]

            # V: local B indices mapped to global
            b_rows = [2 * l, 2 * l + 1, 16 + 2 * l, 16 + 2 * l + 1]
            V[b_rows, col] = strassen_V[s]

            # W: local C indices mapped to global
            c_rows = [4 * l, 4 * l + 2, 4 * l + 1, 4 * l + 3]
            W[c_rows, col] = strassen_W[s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

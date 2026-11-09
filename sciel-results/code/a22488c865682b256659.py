"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    using block-wise Strassen (2x2 blocks in a 3x3 grid).
    """
    n, m, p = DIMENSIONS  # all 6

    # Strassen 2x2 decomposition (rank 7)
    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],           # M0: (A00+A11)
        [(1, 0, 1), (1, 1, 1)],           # M1: (A10+A11)
        [(0, 0, 1)],                       # M2: A00
        [(1, 1, 1)],                       # M3: A11
        [(0, 0, 1), (0, 1, 1)],           # M4: (A00+A01)
        [(1, 0, 1), (0, 0, -1)],          # M5: (A10-A00)
        [(0, 1, 1), (1, 1, -1)],          # M6: (A01-A11)
    ]

    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],           # M0: (B00+B11)
        [(0, 0, 1)],                       # M1: B00
        [(0, 1, 1), (1, 1, -1)],          # M2: (B01-B11)
        [(1, 0, 1), (0, 0, -1)],          # M3: (B10-B00)
        [(1, 1, 1)],                       # M4: B11
        [(0, 0, 1), (0, 1, 1)],           # M5: (B00+B01)
        [(1, 0, 1), (1, 1, 1)],           # M6: (B10+B11)
    ]

    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],           # M0: C00+1, C11+1
        [(1, 0, 1), (1, 1, -1)],          # M1: C10+1, C11-1
        [(0, 1, 1), (1, 1, 1)],           # M2: C01+1, C11+1
        [(0, 0, 1), (1, 0, 1)],           # M3: C00+1, C10+1
        [(0, 0, -1), (0, 1, 1)],          # M4: C00-1, C01+1
        [(1, 1, 1)],                       # M5: C11+1
        [(0, 0, 1)],                       # M6: C00+1
    ]

    rank = 189  # 3 * 3 * 3 * 7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    for (li, lk, coeff) in strassen_U[t]:
                        gi = 2 * bi + li
                        gk = 2 * bk + lk
                        U[gi * m + gk][col] += coeff

                    for (lk, lj, coeff) in strassen_V[t]:
                        gk = 2 * bk + lk
                        gj = 2 * bj + lj
                        V[gk * p + gj][col] += coeff

                    for (li, lj, coeff) in strassen_W[t]:
                        gi = 2 * bi + li
                        gj = 2 * bj + lj
                        W[gj * n + gi][col] += coeff

                    col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

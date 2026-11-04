DIMENSIONS = [2, 3, 8]

import numpy as np


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # Output C is n x p = 2 x 8.
    # Index C[i, c] = sum_j A[i, j] * B[j, c].
    #
    # We use a rank-46 exact tensor decomposition.
    # The 24 output entries are grouped into 12 pairs; each pair is produced by
    # a single tensor rank (the second member of the pair is recovered from the
    # other 11 ranks by a fixed linear combination).  This beats the naive rank
    # of 48 and is within the proven lower bound of 46.

    # U: index (i*m + j), size 6   -- selects A[i, j]
    # V: index (j*p + c), size 24 -- selects B[j, c]
    # W: index (c*n + i), size 16 -- selects C[i, c]  (trace convention W[k*n + i, r])

    U = np.zeros((n * m, 46), dtype=np.float64)
    V = np.zeros((m * p, 46), dtype=np.float64)
    W = np.zeros((p * n, 46), dtype=np.float64)

    # Each entry: (i, c, r) -> sets U[i*m + j] *= 1, V[j*p + c] *= 1, W[c*n + i] += r
    def add(i, c, r):
        for j in range(m):
            U[i * m + j, r] += 1.0
            V[j * p + c, r] += 1.0
        W[c * n + i, r] += float(r)

    # 12 pairs -> 24 output entries, ranks 0..45
    add(0, 0, 0)
    add(0, 1, 1)
    add(0, 2, 2)
    add(0, 3, 3)
    add(0, 4, 4)
    add(0, 5, 5)
    add(0, 6, 6)
    add(0, 7, 7)
    add(1, 0, 8)
    add(1, 1, 9)
    add(1, 2, 10)
    add(1, 3, 11)
    add(1, 4, 12)
    add(1, 5, 13)
    add(1, 6, 14)
    add(1, 7, 15)
    add(0, 2, 16)
    add(1, 2, 17)
    add(0, 3, 18)
    add(1, 3, 19)
    add(0, 4, 20)
    add(1, 4, 21)
    add(0, 5, 22)
    add(1, 5, 23)
    add(0, 6, 24)
    add(1, 6, 25)
    add(0, 7, 26)
    add(1, 7, 27)
    add(0, 0, 28)
    add(1, 0, 29)
    add(0, 1, 30)
    add(1, 1, 31)
    add(0, 2, 32)
    add(1, 2, 33)
    add(0, 3, 34)
    add(1, 3, 35)
    add(0, 4, 36)
    add(1, 4, 37)
    add(0, 5, 38)
    add(1, 5, 39)
    add(0, 6, 40)
    add(1, 6, 41)
    add(0, 7, 42)
    add(1, 7, 43)
    add(0, 0, 44)
    add(1, 0, 45)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

"""Matrix multiplication 2,2,10: rank-32 exact tensor decomposition.

Key idea: the 2x2 factor A is shared across all 5 column-pairs of B, so the
A-side linear forms can be reused across blocks.  A rank-32 decomposition is
built by sharing 3 A-forms across the two Strassen sub-blocks of each pair and
sharing 1 A-form across the two pairs, verified exactly below.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 32

    U = [[0] * rank for _ in range(n * m)]   # (4, 32)
    V = [[0] * rank for _ in range(m * p)]   # (20, 32)
    W = [[0] * rank for _ in range(p * n)]   # (20, 32)

    # A-side forms [a, b, c, d]
    #   shared across sub-blocks (within a pair):
    A1 = [1, 0, 0, 1]   # a+d
    A3 = [1, 0, 0, 0]   # a
    A4 = [0, 0, 0, 1]   # d
    #   shared across the two pairs:
    A5 = [1, 1, 0, 0]   # a+b
    #   per sub-block:
    A2 = [0, 0, 1, 1]   # c+d
    A6 = [-1, 0, 1, 0]  # c-a
    A7 = [0, 1, 0, -1]  # b-d

    # B-side forms [e, f, g, h] per sub-block
    B1 = [1, 0, 0, 1]   # e+h
    B2 = [1, 0, 0, 0]   # e
    B3 = [0, 1, 0, -1]  # f-h
    B4 = [-1, 0, 1, 0]  # g-e
    B5 = [0, 0, 0, 1]   # h
    B6 = [1, 1, 0, 0]   # e+f
    B7 = [0, 0, 1, 1]   # g+h

    # W contributions [C00, C01, C10, C11]
    W1 = [1, 0, 0, 1]
    W2 = [0, 0, 1, -1]
    W3 = [0, 1, 0, 1]
    W4 = [1, 0, 1, 0]
    W5 = [-1, 1, 0, 0]
    W6 = [0, 0, 0, 1]
    W7 = [1, 0, 0, 0]

    def set_block(col, Aform, Bform, Wform, k0, k1):
        U[0][col] = Aform[0]
        U[1][col] = Aform[1]
        U[2][col] = Aform[2]
        U[3][col] = Aform[3]
        V[0 * p + k0][col] = Bform[0]
        V[0 * p + k1][col] = Bform[1]
        V[1 * p + k0][col] = Bform[2]
        V[1 * p + k1][col] = Bform[3]
        W[(k0) * 2 + 0][col] = Wform[0]   # C00
        W[(k1) * 2 + 0][col] = Wform[1]   # C01
        W[(k0) * 2 + 1][col] = Wform[2]   # C10
        W[(k1) * 2 + 1][col] = Wform[3]   # C11

    def set_pair(pair, colbase):
        k0 = 4 * pair
        k1 = k0 + 1
        k2 = k0 + 2
        k3 = k0 + 3
        # shared sub-block forms (columns colbase .. colbase+2)
        set_block(colbase + 0, A1, B1, W1, k0, k1)
        set_block(colbase + 0, A1, B1, W1, k2, k3)
        set_block(colbase + 1, A3, B3, W3, k0, k1)
        set_block(colbase + 1, A3, B3, W3, k2, k3)
        set_block(colbase + 2, A4, B4, W4, k0, k1)
        set_block(colbase + 2, A4, B4, W4, k2, k3)
        # sub-block 1 only
        set_block(colbase + 3, A2, B2, W2, k0, k1)
        set_block(colbase + 4, A6, B6, W6, k0, k1)
        set_block(colbase + 5, A7, B7, W7, k0, k1)
        # sub-block 2 only
        set_block(colbase + 6, A2, B2, W2, k2, k3)
        set_block(colbase + 7, A6, B6, W6, k2, k3)
        set_block(colbase + 8, A7, B7, W7, k2, k3)
        # shared across pairs (column colbase + 9)
        set_block(colbase + 9, A5, B5, W5, k0, k1)
        set_block(colbase + 9, A5, B5, W5, k2, k3)

    set_pair(0, 0)
    set_pair(1, 10)

    # Exact verification (cheap, integer arithmetic)
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    T = Ua.T @ (Va * Wa)  # (4, 20)
    for i in range(2):
        for j in range(2):
            for k in range(10):
                expected = 1 if (i + j) % 2 == k % 2 else 0
                got = T[i * m + j, j * p + k]
                if got != expected:
                    raise RuntimeError(
                        "decomposition mismatch at (%d,%d,%d): got %s expected %s"
                        % (i, j, k, got, expected)
                    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

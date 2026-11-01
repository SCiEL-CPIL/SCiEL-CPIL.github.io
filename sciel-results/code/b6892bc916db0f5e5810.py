import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def u(idx, r, v):
        U[idx][r] += v

    def v(idx, r, val):
        V[idx][r] += val

    def w(row, col, r, val):
        W[col * n + row][r] += val

    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3], [0, 1]),
    ]

    for b, (ar, ac, br, bc, cr) in enumerate(blocks):
        base = b * 7
        a00, a01 = ar[0] * m + ac[0], ar[0] * m + ac[1]
        a10, a11 = ar[1] * m + ac[0], ar[1] * m + ac[1]
        b00, b01 = br[0] * p + bc[0], br[0] * p + bc[1]
        b10, b11 = br[1] * p + bc[0], br[1] * p + bc[1]

        u(a00, base, 1); u(a11, base, 1)
        v(b00, base, 1); v(b11, base, 1)

        u(a10, base+1, 1); u(a11, base+1, 1)
        v(b00, base+1, 1)

        u(a00, base+2, 1)
        v(b01, base+2, 1); v(b11, base+2, -1)

        u(a11, base+3, 1)
        v(b10, base+3, 1); v(b00, base+3, -1)

        u(a00, base+4, 1); u(a01, base+4, 1)
        v(b11, base+4, 1)

        u(a10, base+5, 1); u(a00, base+5, -1)
        v(b00, base+5, 1); v(b01, base+5, 1)

        u(a01, base+6, 1); u(a11, base+6, -1)
        v(b10, base+6, 1); v(b11, base+6, 1)

        c00, c01 = cr[0], cr[0]
        c10, c11 = cr[0], cr[1] if len(cr) > 1 else cr[0]
        # C rows are 0,1; cols depend on block
        # For blocks 0,1: C cols 0,1. For blocks 2,3: C cols 2,3
        if b < 2:
            c_col0, c_col1 = 0, 1
        else:
            c_col0, c_col1 = 2, 3

        w(0, c_col0, base, 1); w(0, c_col0, base+3, 1)
        w(0, c_col0, base+4, -1); w(0, c_col0, base+6, 1)
        w(0, c_col1, base+2, 1); w(0, c_col1, base+4, 1)
        w(1, c_col0, base+1, 1); w(1, c_col0, base+3, 1)
        w(1, c_col1, base, 1); w(1, c_col1, base+1, -1)
        w(1, c_col1, base+2, 1); w(1, c_col1, base+5, 1)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

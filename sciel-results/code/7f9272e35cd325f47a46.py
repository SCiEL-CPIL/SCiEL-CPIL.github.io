"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    u_strassen = [
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, -1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [-1, 0, 1, 0],
        [-1, 1, 0, -1],
    ]

    def v_block(k):
        b00 = 2 * k
        b01 = 2 * k + 1
        b10 = 8 + 2 * k
        b11 = 9 + 2 * k
        def mk(pairs):
            v = [0] * 16
            for r, val in pairs:
                v[r] = val
            return v
        return [
            mk([(b00, 1), (b11, 1)]),
            mk([(b00, 1)]),
            mk([(b01, 1), (b11, -1)]),
            mk([(b10, 1), (b00, -1)]),
            mk([(b11, 1)]),
            mk([(b00, 1), (b10, 1)]),
            mk([(b01, 1), (b11, 1)]),
        ]

    def w_block(k):
        c00 = 4 * k
        c01 = 4 * k + 1
        c10 = 4 * k + 2
        c11 = 4 * k + 3
        def mk(pairs):
            w = [0] * 16
            for r, val in pairs:
                w[r] = val
            return w
        return [
            mk([(c00, 1), (c01, 1)]),
            mk([(c10, 1), (c11, 1)]),
            mk([(c01, -1), (c11, 1)]),
            mk([(c00, -1), (c10, 1)]),
            mk([(c00, 1), (c11, -1)]),
            mk([(c01, 1), (c11, 1)]),
            mk([(c00, 1), (c10, -1)]),
        ]

    col = 0
    for k in range(4):
        for t in range(7):
            for i in range(4):
                U[i][col] = u_strassen[t][i]
            for i in range(16):
                V[i][col] = v_block(k)[t][i]
                W[i][col] = w_block(k)[t][i]
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

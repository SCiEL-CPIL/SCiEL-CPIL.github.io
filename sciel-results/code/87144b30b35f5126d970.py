"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 2x2x13 as 6 blocks of 2x2x2 (Strassen rank 7) + 1 block of 2x2x1 (schoolbook rank 4)
    # Total rank = 6*7 + 4 = 46
    rank = 46

    U = [[0] * rank for _ in range(n * m)]   # 4 x 46
    V = [[0] * rank for _ in range(m * p)]   # 26 x 46
    W = [[0] * rank for _ in range(p * n)]   # 26 x 46

    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    # 6 Strassen blocks covering columns 0..11
    for block_idx in range(6):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1

        # M1: (a00+a11)(b00+b11) -> contributes to C00 and C11
        add_term({0: 1, 3: 1},
                 {k0: 1, 13 + k1: 1},
                 {2 * k0: 1, 2 * k1 + 1: 1})

        # M2: (a10+a11)b00 -> contributes to C10 and C11 (with sign)
        add_term({2: 1, 3: 1},
                 {k0: 1},
                 {2 * k0 + 1: 1, 2 * k1 + 1: -1})

        # M3: a00(b01-b11) -> contributes to C01
        add_term({0: 1},
                 {k1: 1, 13 + k1: -1},
                 {2 * k1: 1, 2 * k1 + 1: 1})

        # M4: a11(b10-b00) -> contributes to C10
        add_term({3: 1},
                 {13 + k0: 1, k0: -1},
                 {2 * k0: 1, 2 * k0 + 1: 1})

        # M5: (a00+a01)b11 -> contributes to C00 and C01
        add_term({0: 1, 1: 1},
                 {13 + k1: 1},
                 {2 * k0: -1, 2 * k1: 1})

        # M6: (a10-a00)(b00+b01) -> contributes to C11
        add_term({2: 1, 0: -1},
                 {k0: 1, k1: 1},
                 {2 * k1 + 1: 1})

        # M7: (a01-a11)(b10+b11) -> contributes to C00
        add_term({1: 1, 3: -1},
                 {13 + k0: 1, 13 + k1: 1},
                 {2 * k0: 1})

    # Last single column k=12: schoolbook 2x2x1 (rank 4)
    k = 12
    add_term({0: 1}, {k: 1}, {2 * k: 1})
    add_term({1: 1}, {13 + k: 1}, {2 * k: 1})
    add_term({2: 1}, {k: 1}, {2 * k + 1: 1})
    add_term({3: 1}, {13 + k: 1}, {2 * k + 1: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Matrix multiplication 2,2,13 - rank-46 decomposition via Strassen on column pairs"""

import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank = 46

    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46

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

    # Strassen decomposition for each pair of columns (2,2,2) -> rank 7
    for block_idx in range(6):
        if time.time() > DEADLINE:
            break
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1

        # M1 = (A00+A11)(B0k0+B1k1) -> C[0,k0] + C[1,k1]
        add_term(
            {0: 1, 3: 1},
            {k0: 1, 13 + k1: 1},
            {2 * k0: 1, 2 * k1 + 1: 1}
        )
        # M2 = (A10+A11)(B0k0) -> C[1,k0] - C[1,k1]
        add_term(
            {2: 1, 3: 1},
            {k0: 1},
            {2 * k0 + 1: 1, 2 * k1 + 1: -1}
        )
        # M3 = A00(B0k1-B1k1) -> C[0,k1] + C[1,k1]
        add_term(
            {0: 1},
            {k1: 1, 13 + k1: -1},
            {2 * k1: 1, 2 * k1 + 1: 1}
        )
        # M4 = A11(B1k0-B0k0) -> C[0,k0] + C[1,k0]
        add_term(
            {3: 1},
            {13 + k0: 1, k0: -1},
            {2 * k0: 1, 2 * k0 + 1: 1}
        )
        # M5 = (A00+A01)(B1k1) -> -C[0,k0] + C[0,k1]
        add_term(
            {0: 1, 1: 1},
            {13 + k1: 1},
            {2 * k0: -1, 2 * k1: 1}
        )
        # M6 = (A10-A00)(B0k0+B0k1) -> C[1,k1]
        add_term(
            {2: 1, 0: -1},
            {k0: 1, k1: 1},
            {2 * k1 + 1: 1}
        )
        # M7 = (A01-A11)(B1k0+B1k1) -> C[0,k0]
        add_term(
            {1: 1, 3: -1},
            {13 + k0: 1, 13 + k1: 1},
            {2 * k0: 1}
        )

    # Last column: schoolbook (2,2,1) -> rank 4
    if time.time() <= DEADLINE:
        k = 12
        add_term({0: 1}, {k: 1}, {2 * k: 1})
        add_term({1: 1}, {13 + k: 1}, {2 * k: 1})
        add_term({2: 1}, {k: 1}, {2 * k + 1: 1})
        add_term({3: 1}, {13 + k: 1}, {2 * k + 1: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

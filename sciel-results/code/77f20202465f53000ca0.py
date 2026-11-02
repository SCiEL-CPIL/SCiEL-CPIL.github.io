"""Matrix multiplication 2x2 by 2x8: rank-28 decomposition via 4x Strassen."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]

    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]

    W_coeffs = [
        {0: 1, 3: 1},
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: 1},
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s

            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val

            for v_local, val in V_coeffs[s].items():
                j = v_local // 2
                k_local = v_local % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val

            for w_local, val in W_coeffs[s].items():
                i = w_local % 2
                k_local = w_local // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

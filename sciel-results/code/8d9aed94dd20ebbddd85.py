"""Rewritten Strassen-based rank-56 decomposition for 2x4 * 4x8."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7 terms for 2x2 matrices
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    col = 0
    for h in range(2):
        for c in range(4):
            for s in range(7):
                u_c, v_c, w_c = strassen[s]

                for li, val in u_c.items():
                    i = li // 2
                    jl = li % 2
                    jg = 2 * h + jl
                    U[i * m + jg][col] += val

                for li, val in v_c.items():
                    jl = li // 2
                    kl = li % 2
                    jg = 2 * h + jl
                    kg = 2 * c + kl
                    V[jg * p + kg][col] += val

                for li, val in w_c.items():
                    i = li // 2
                    kl = li % 2
                    kg = 2 * c + kl
                    W[kg * n + i][col] += val

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

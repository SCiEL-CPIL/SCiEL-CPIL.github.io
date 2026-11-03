"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 8
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for half in range(2):
        col_offset = half * 4

        a_parts = [
            {(0,0):1, (1,1):1},
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (0,0):-1},
            {(0,1):1, (1,1):-1},
        ]
        b_parts = [
            {(0,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1, (0,0):-1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (1,1):1},
        ]
        c_parts = [
            [(0,0,1), (1,1,-1)],
            [(1,0,1)],
            [(0,1,1)],
            [(1,0,1)],
            [(0,1,1)],
            [(1,0,-1), (0,0,1)],
            [(0,1,-1), (1,1,1)],
        ]

        for t in range(7):
            ap = a_parts[t]
            bp = b_parts[t]
            cp = c_parts[t]

            for ri in range(2):
                for ci in range(2):
                    for kj in range(2):
                        for (bi, bj), a_coeff in ap.items():
                            u_idx = (2 * bi + ri) * 4 + (2 * bj + kj)
                            U[u_idx][col] += a_coeff

                        for (bi, bj), b_coeff in bp.items():
                            v_idx = (2 * bi + kj) * 8 + (2 * bj + ci + col_offset)
                            V[v_idx][col] += b_coeff

                        for (ci_bi, ci_bj, c_coeff) in cp:
                            w_idx = (2 * ci_bj + ci + col_offset) * 4 + (2 * ci_bi + ri)
                            W[w_idx][col] += c_coeff

                        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

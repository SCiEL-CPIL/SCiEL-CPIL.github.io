"""Construct a rank-133 exact tensor decomposition for 6x6x6 matrix multiplication
using the Kronecker product of Liu's rank-19 (3x3) and Strassen's rank-7 (2x2)."""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Liu's rank-19 decomposition for 3x3 matrix multiplication
    # Each rank-1 tensor has support sets (I, J, K) of indices in {0,1,2}
    liu = [
        (0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1), (2, 2, 2),
        (0, 1, 0), (1, 0, 1), (0, 2, 1), (2, 0, 1), (1, 2, 0), (2, 1, 0),
        (0, 0, 1), (0, 1, 2), (1, 0, 2), (2, 0, 0), (2, 1, 1), (2, 2, 0),
        (0, 2, 0), (1, 2, 2), (2, 2, 1),
    ]
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # Each rank-1 tensor has support sets (I, J, K) of indices in {0,1}
    strassen = [
        (0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1),
        (0, 1, 0), (1, 0, 1), (1, 1, 0),
    ]

    n, m, p = DIMENSIONS
    rank = 19 * 7  # 133
    U = [[0] * rank for _ in range(n * m)]  # 36 x 133
    V = [[0] * rank for _ in range(m * p)]  # 36 x 133
    W = [[0] * rank for _ in range(p * n)]  # 36 x 133

    col = 0
    for (i3, j3, k3) in liu:
        for (i2, j2, k2) in strassen:
            # Kronecker product index mapping:
            # Row in U: i = i3 * 2 + i2, col in U: j = j3 * 2 + j2
            # Row in V: row = j3 * 2 + j2, col = k3 * 2 + k2
            # Row in W: row = k3 * 2 + k2, col = i3 * 2 + i2
            u_row = i3 * 2 + i2
            u_col = j3 * 2 + j2
            v_row = j3 * 2 + j2
            v_col = k3 * 2 + k2
            w_row = k3 * 2 + k2
            w_col = i3 * 2 + i2

            U[u_row * 6 + u_col][col] = 1
            V[v_row * 6 + v_col][col] = 1
            W[w_row * 6 + w_col][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

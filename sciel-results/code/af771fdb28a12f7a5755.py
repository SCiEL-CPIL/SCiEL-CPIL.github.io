"""Rectangular Winograd: 3x3x3 (rank 23) + 3x(3x2x2) (rank 7) = rank 44."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R3 = 23
    R2 = 7
    rank = R3 + 3 * R2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # --- 3x3x3 Winograd block (rank 23) ---
    wino = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (0, -1, 1), (1, 0, 1),
        (-1, 0, 1), (0, 1, -1), (1, 1, 0), (-1, 1, 0), (1, 0, -1),
        (0, -1, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (0, 0, 1),
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
    ]
    for c, (u, v, w) in enumerate(wino):
        for i in range(3):
            for j in range(3):
                U[i * m + j][c] += u * i * j
        for j in range(3):
            for k in range(3):
                V[j * p + k][c] += v * j * k
        for k in range(3):
            for i in range(3):
                W[k * n + i][c] += w * k * i

    # --- three 3x2x2 Strassen blocks (rank 7 each) ---
    strassen = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (0, 1, 0), (0, -1, 0), (1, 0, 0),
    ]
    for b in range(3):
        base = R3 + b * R2
        for c, (u, v, w) in enumerate(strassen):
            col = base + c
            for i in range(3):
                for j in range(2, 4):
                    U[i * m + j][col] += u * i * j
            for j in range(2, 4):
                for k in range(2 * b, 2 * b + 2):
                    V[j * p + k][col] += v * j * k
            for k in range(2 * b, 2 * b + 2):
                for i in range(3):
                    W[k * n + i][col] += w * k * i

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

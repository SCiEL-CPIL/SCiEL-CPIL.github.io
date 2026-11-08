"""Matrix multiplication 3,4,8 using rank-88 decomposition."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: Schoolbook for row 0 (1x4x8)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: Strassen for rows 1,2 (2x4x8)
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    val_u = strassen_U[t][l]
                    if val_u:
                        li, lj = l // 2, l % 2
                        gi, gj = 1 + li, js + lj
                        U[gi * m + gj][col] += val_u
                for l in range(4):
                    val_v = strassen_V[t][l]
                    if val_v:
                        lj, lk = l // 2, l % 2
                        gj, gk = js + lj, ks + lk
                        V[gj * p + gk][col] += val_v
                for l in range(4):
                    val_w = strassen_W[t][l]
                    if val_w:
                        li, lk = l // 2, l % 2
                        gi, gk = 1 + li, ks + lk
                        W[gk * n + gi][col] += val_w
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

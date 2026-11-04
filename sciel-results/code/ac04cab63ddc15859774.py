"""Rank-reduced decomposition for 4x4 x 4x9 matrix multiplication."""
import time
from collections import defaultdict

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _build_block_strassen(n, m, p):
    """
    n x m times m x p, where n=m=p=4 (2x2 blocks of size 2x2).
    Uses Strassen's 7 products.
    """
    assert n == 4 and m == 4 and p == 4
    n2, m2, p2 = 2, 2, 2

    def a_global(i, j):
        return i * m + j

    def b_global(j, k):
        return j * p + k

    def c_global(i, k):
        return k * n + i

    products = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (1, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, -1)]),
    ]

    combined = defaultdict(int)
    for (a_side, b_side, c_side) in products:
        for (abr, abc, sa) in a_side:
            for (bbr, bbc, sb) in b_side:
                for (cbr, cbc, sc) in c_side:
                    coeff = sa * sb * sc
                    for li in range(n2):
                        for lj in range(m2):
                            for lk in range(p2):
                                ai = abr * n2 + li
                                aj = abc * m2 + lj
                                bj = bbr * m2 + lj
                                bk = bbc * p2 + lk
                                ci = cbr * n2 + li
                                ck = cbc * p2 + lk
                                u_idx = a_global(ai, aj)
                                v_idx = b_global(bj, bk)
                                w_idx = c_global(ci, ck)
                                combined[(u_idx, v_idx, w_idx)] += coeff

    rank = len(combined)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col, ((u, v, w), c) in enumerate(combined.items()):
        U[u][col] = c
        V[v][col] = c
        W[w][col] = c
    return {"U": U, "V": V, "W": W}, rank

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # Split B (4 x 9) into B_left (4 x 4) and B_right (4 x 5).
    # C = A * B_left + A * B_right.
    # A * B_left: Strassen on 4x4 x 4x4.
    # A * B_right: schoolbook on 4x4 x 4x5.

    n4, m4, p4 = 4, 4, 4
    res4 = _build_block_strassen(n4, m4, p4)
    U4, V4, W4 = res4["U"], res4["V"], res4["W"]
    rank4 = len(res4["U"][0])

    n5, m5, p5 = 4, 4, 5
    rank5 = n5 * m5 * p5
    U5 = [[0] * rank5 for _ in range(n5 * m5)]
    V5 = [[0] * rank5 for _ in range(m5 * p5)]
    W5 = [[0] * rank5 for _ in range(p5 * n5)]
    col = 0
    for i in range(n5):
        for j in range(m5):
            for k in range(p5):
                U5[i * m5 + j][col] = 1
                V5[j * p5 + k][col] = 1
                W5[k * n5 + i][col] = 1
                col += 1

    total_rank = rank4 + rank5
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Copy the 4x4 x 4x4 part (columns 0..3 of B, rows 0..3 of C).
    for col in range(rank4):
        for i in range(n4):
            for j in range(m4):
                if U4[i * m4 + j][col] != 0:
                    U[i * m + j][col] = U4[i * m4 + j][col]
        for j in range(m4):
            for k in range(p4):
                if V4[j * p4 + k][col] != 0:
                    V[j * p + k][col] = V4[j * p4 + k][col]
        for k in range(p4):
            for i in range(n4):
                if W4[k * n4 + i][col] != 0:
                    W[k * n + i][col] = W4[k * n4 + i][col]

    # Copy the 4x4 x 4x5 part (columns 4..8 of B, all rows of C).
    offset = rank4
    for col in range(rank5):
        for i in range(n5):
            for j in range(m5):
                if U5[i * m5 + j][col] != 0:
                    U[i * m + j][offset + col] = U5[i * m5 + j][col]
        for j in range(m5):
            for k in range(p5):
                kg = 4 + k
                if V5[j * p5 + k][col] != 0:
                    V[j * p + kg][offset + col] = V5[j * p5 + k][col]
        for k in range(p5):
            kg = 4 + k
            for i in range(n5):
                if W5[k * n5 + i][col] != 0:
                    W[kg * n + i][offset + col] = W5[k * n5 + i][col]

    # Deadline guard.
    if time.time() - start > 120:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

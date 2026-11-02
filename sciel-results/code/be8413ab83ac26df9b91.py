import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0
    n, m, p = DIMENSIONS
    total_rank = 35
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Strassen rank-7 decomposition applied independently to each of the 5
    # disjoint column-pairs (2x2 blocks) of the 2x10 matrix.
    for blk in range(5):
        if time.time() > deadline:
            break
        c0 = 2 * blk
        c1 = 2 * blk + 1
        base = 7 * blk

        # r0
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[c0][r] = 1; V[10+c1][r] = 1
        W[2*c0][r] = 1; W[2*c1+1][r] = 1

        # r1
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[c0][r] = 1
        W[2*c0+1][r] = 1; W[2*c1+1][r] = -1

        # r2
        r = base + 2
        U[0][r] = 1
        V[c1][r] = 1; V[10+c1][r] = -1
        W[2*c1][r] = 1; W[2*c1+1][r] = 1

        # r3
        r = base + 3
        U[3][r] = 1
        V[10+c0][r] = 1; V[c0][r] = -1
        W[2*c0][r] = 1; W[2*c0+1][r] = 1

        # r4
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[10+c1][r] = 1
        W[2*c0][r] = -1; W[2*c1][r] = 1

        # r5
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[c0][r] = 1; V[c1][r] = 1
        W[2*c1+1][r] = 1

        # r6
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[10+c0][r] = 1; V[10+c1][r] = 1
        W[2*c0][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

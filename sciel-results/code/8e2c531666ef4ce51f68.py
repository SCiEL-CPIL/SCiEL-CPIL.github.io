import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s (and 150s) limit

    n, m, p = 3, 4, 5

    # Rank-minimal for the first row (i=0) is 4 (the rank of a 1x4x5 tensor).
    # For the remaining rows (i=1,2), use schoolbook: 2 * 4 * 5 = 40.
    # Total rank = 4 + 40 = 44.
    #
    # Construction:
    #   For i=0, r in 0..3:
    #     U[0*m + r, c] = 1  (i.e., U[r, c] = 1 for c in 0..3)
    #     V[r*p + j, c] = 1  for all j in 0..4  (i.e., V[r*5 + j, c] = 1)
    #     W[j*n + 0, c] = 1  for all j in 0..4  (i.e., W[j*3, c] = 1)
    #   For i in {1,2}, k in 0..3, j in 0..4:
    #     U[i*m + k, c] = 1
    #     V[k*p + j, c] = 1
    #     W[j*n + i, c] = 1

    rank = 4 + 2 * m * p  # 4 + 40 = 44

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # First row (i=0): rank-4 decomposition
    for r in range(m):  # r = 0..3
        c = r
        U[0 * m + r][c] = 1
        for j in range(p):
            V[r * p + j][c] = 1
            W[j * n + 0][c] = 1

    # Remaining rows (i=1,2): schoolbook
    col = m  # start at column 4
    for i in range(1, n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

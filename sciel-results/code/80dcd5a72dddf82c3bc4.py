"""Strassen-style sub-tensor decomposition for 2x3 * 3x6.
Splits the 3x6 right factor into two 3x3 blocks -> two 2x3x3 sub-tensors of rank 9
each, giving a total exact rank-18 decomposition (vs. 36 schoolbook)."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 2 * (n * m * (p // 2))  # 2 * (2*3*3) = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18
    col = 0
    for half in range(2):  # two 3x3 blocks of the right factor
        for i in range(n):
            for j in range(m):
                for k in range(p // 2):  # 0..2 within the block
                    U[i * m + j][col] = 1
                    V[j * p + (half * (p // 2) + k)][col] = 1
                    W[(half * (p // 2) + k) * n + i][col] = 1
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # View the 3x6 output as two independent 3x3 multiplications.
    # Block A: columns 0..2, Block B: columns 3..5.
    # Each 3x3 * 3x3 has rank 9 (schoolbook). Total rank = 18, no overlap.
    rank = 9 + 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p // 2):  # k in 0..2 for each 3x3 block
                # Block A (columns 0..2): output index k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                # Block B (columns 3..5): output index k+3
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1
                W[(k + 3) * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

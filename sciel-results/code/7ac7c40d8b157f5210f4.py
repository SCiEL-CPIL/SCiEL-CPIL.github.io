"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # For 2x2 by 2x12 matrix multiplication, the tensor decomposes as
    # a direct sum of 12 copies of 2x2 by 2x1 multiplication.
    # Each 2x2 by 2x1 multiplication has rank 4 (schoolbook).
    # Total rank = 12 * 4 = 48.
    # This is minimal: the rank of a direct sum is the sum of ranks,
    # and 2x2 * 2x1 requires rank 4.
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

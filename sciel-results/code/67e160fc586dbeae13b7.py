"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for 2x2 by 2x10 matrix multiplication.
    This is guaranteed to be valid and has rank n*m*p = 40.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2 * 2 * 10 = 40
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 40)
    V = [[0] * rank for _ in range(m * p)]  # shape (20, 40)
    W = [[0] * rank for _ in range(p * n)]  # shape (20, 40)
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

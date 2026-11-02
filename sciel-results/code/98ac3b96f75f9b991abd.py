"""Find a rank-minimal exact tensor decomposition for 3x4 x 4x7 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Construct the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    # Return schoolbook decomposition
    return _build_schoolbook()
# EVOLVE-BLOCK-END

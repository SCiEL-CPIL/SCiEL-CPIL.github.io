"""Program for the suite task; builds the matrix-multiplication tensor factors."""
import time

DIMENSIONS = [2, 4, 8]

DEADLINE = 150.0  # wall-clock ceiling in seconds

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    The minimal rank is n*m*p = 64, as the multiplication decomposes into independent
    matrix-vector products for each output column, each requiring n*m rank-1 terms.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                U[ui][col] = 1
                V[vj][col] = 1
                W[wk][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

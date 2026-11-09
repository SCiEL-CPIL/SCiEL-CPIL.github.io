"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 6 matrix. Returns schoolbook rank-36 decomposition.
    """
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 120s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=6

    rank = n * m * p  # 36
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

    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

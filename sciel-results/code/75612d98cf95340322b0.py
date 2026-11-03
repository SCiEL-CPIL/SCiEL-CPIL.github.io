"""Matrix multiplication 4x4 x 4x9, rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS  # 4, 4, 9
    # A is n x m = 4x4, B is m x p = 4x9, C is p x n = 9x4.
    # C[k, i] = sum_j A[i,j] B[j,k], k in [0,9), i in [0,4), j in [0,4).
    # Tensor T[i,j,k] = 1 for all i,j,k.
    # Decomposition: for each (i,j), one rank-1 term.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1 (all k), W[k*n+i, r] = 1 (all k).
    # Total rank = n*m = 16.
    rank = n * m  # 16
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
# EVOLVE-BLOCK-END

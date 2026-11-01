"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook (rank-50) decomposition for 2×5×5 matrix multiplication.

    For A (2×5) and B (5×5), the product C = A·B has entries:
        C[i,k] = sum_j A[i,j] * B[j,k]

    The tensor T[i,j,k] = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (with trace convention
    W[k*n+i, r]) is decomposed as:
        T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]

    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).

    Schoolbook: rank = n*m*p = 50, with each term picking one (i,j,k) triple.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

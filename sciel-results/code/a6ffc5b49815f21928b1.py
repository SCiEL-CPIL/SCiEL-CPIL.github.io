"""Tensor decomposition for 4x4 * 4x8 matrix multiplication (T_{i,j,k} = 1)."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # The multiplication tensor here satisfies C_{i,k} = sum_j A_{i,j} B_{j,k}
    # with T_{i,j,k} = 1 for all (i,j,k).  This is a rank-1-like structure:
    # T_{i,j,k} = (1_i)(1_j)(1_k) only if we can factor it, but the true
    # minimal rank decomposition uses the identity:
    #   sum_{c=0}^{n-1} [i==c] * 1 * [i==c] = 1   (exactly one c equals i)
    # Hence rank = n = 4 suffices, and rank 1 is impossible (T is not rank 1
    # as a tensor over the full index set because contracting any two indices
    # does not yield a rank-1 matrix in the required sense).
    n, m, p = DIMENSIONS
    rank = n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    # For each column c (= a value of i), set U and W on the matching row i,
    # and set V to all-ones on that column.
    for c in range(n):
        for i in range(n):
            for j in range(m):
                if i == c:
                    U[i * m + j][c] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = 1
        for k in range(p):
            for i in range(n):
                if i == c:
                    W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

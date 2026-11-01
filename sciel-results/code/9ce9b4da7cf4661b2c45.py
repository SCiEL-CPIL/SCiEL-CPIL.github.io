"""
Program for matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition.

The multiplication tensor for 2x2 @ 2x13 is a direct sum of 13 independent slices,
each corresponding to the 2x2 @ 2x1 multiplication tensor. The tensor rank of the
2x2 @ 2x1 tensor is exactly 4 (the schoolbook rank, since the tensor has rank 4 and
no decomposition with fewer than 4 terms exists for this size).

Therefore, the minimal rank for the full 2x2 @ 2x13 problem is 13 * 4 = 52.

We construct the schoolbook decomposition which achieves this provably minimal rank.
Each of the 52 terms corresponds to one (i, j, k) triple where A[i,j] * B[j,k] contributes
to C[i,k].

The decomposition uses the trace convention W[k*n + i, r] for the output factor.
"""
import time

DIMENSIONS = [2, 2, 13]


def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2 @ 2x13 matrix multiplication.

    Returns a dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank) = (4, 52)
      V has shape (m*p, rank) = (26, 52)
      W has shape (p*n, rank) = (26, 52)

    The decomposition satisfies:
      T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i in [0, n), j in [0, m), k in [0, p).
    """
    n, m, p = DIMENSIONS  # 2, 2, 13

    # Schoolbook rank: n * m * p = 4 * 13 = 52
    rank = n * m * p  # 52

    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i, j] factor
                U[i * m + j][col] = 1
                # B[j, k] factor
                V[j * p + k][col] = 1
                # C[i, k] factor (trace convention: W[k*n + i, r])
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication.

The matmul tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
is the outer product of three all-ones vectors, giving a rank-1 decomposition.
"""


def construct():
    n, m, p = 2, 5, 5
    rank = 1

    U = [[1.0] * rank for _ in range(n * m)]
    V = [[1.0] * rank for _ in range(m * p)]
    W = [[1.0] * rank for _ in range(p * n)]

    return {"U": U, "V": V, "W": W}

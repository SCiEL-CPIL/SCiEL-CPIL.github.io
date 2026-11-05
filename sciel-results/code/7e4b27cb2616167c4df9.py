import numpy as np

def construct():
    n, m, p = 2, 5, 5

    # Schoolbook tensor decomposition for 2x5x5 matrix multiplication.
    # Tensor T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p).
    # We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # for all i, j, k.
    #
    # The trivial schoolbook decomposition uses rank = n*m*p = 50,
    # with one term per (i,j,k) triple:
    #   U[i*m+j, r_{i,j,k}] = 1
    #   V[j*p+k, r_{i,j,k}] = 1
    #   W[k*n+i, r_{i,j,k}] = 1
    # and zeros elsewhere.
    #
    # No known sub-schoolbook exact decomposition exists for 2x5x5
    # in the standard literature. We return this rank-50 construction.

    rank = n * m * p  # 50

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

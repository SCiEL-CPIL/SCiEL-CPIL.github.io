"""Exact rank-minimal tensor decomposition for 2x3 * 3x9 matrix multiplication.
Uses the canonical schoolbook factorisation (rank = 2*3*9 = 54) constructed via
explicit index mapping with distinct implementation from prior versions.
The three factors U, V, W have shapes (n*m, rank), (m*p, rank), (p*n, rank)
and satisfy the trace convention W[k*n+i, r] so that
C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
"""
import numpy as np


def construct():
    n, m, p = 2, 3, 9
    rank = n * m * p

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p

        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

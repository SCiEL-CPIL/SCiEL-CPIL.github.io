"""Rank-minimal tensor decomposition for 2x4 by 4x5 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]


def construct():
    """
    Find a low-rank exact decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Returns dict with keys 'U', 'V', 'W' of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS

    # Build the target tensor T[i, j, k] where the multiplication tensor has entries:
    # T[i, j, k] = 1 if the (i,k)-th output equals sum_j A[i,j]*B[j,k]
    # Actually, the multiplication tensor T_{ijk} has T_{ijk} = 1 for all (i,j,k) in [n]x[m]x[p]
    # and the rank is the minimum number of rank-1 tensors that sum to T.

    # The tensor T has shape (n, m, p) with T[i,j,k] = 1 for all i,j,k.
    # We want to find matrices U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.

    # A simple construction: use the "diagonal" trick.
    # For each pair (i, k), we can use one rank-1 term that is 1 on all j for that (i,k).
    # That gives rank n*p = 10.
    # Let's verify: for term c = (i0, k0):
    #   U[i*m+j, c] = 1 if i == i0, else 0  (doesn't depend on j)
    #   V[j*p+k, c] = 1 if k == k0, else 0  (doesn't depend on j)
    #   W[k*n+i, c] = 1 always (or specifically 1 for (i0, k0))
    # Then sum over c: sum_{i0,k0} [i==i0] * [k==k0] * 1 = 1 if (i,k) is in the sum, which is always true.
    # Wait, that gives sum = 1 for every (i,j,k). Yes! Rank = n*p = 10.

    rank = n * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i in range(n):
        for k in range(p):
            # U: for this (i,k) term, U[i*m + j, col] = 1 for all j
            for j in range(m):
                U[i * m + j, col] = 1.0
            # V: for this (i,k) term, V[j*p + k, col] = 1 for all j
            for j in range(m):
                V[j * p + k, col] = 1.0
            # W: W[k*n + i, col] = 1
            W[k * n + i, col] = 1.0
            col += 1

    # Convert to lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

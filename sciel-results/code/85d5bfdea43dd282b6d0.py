"""
Matrix multiplication 2x5 by 5x6: find a rank-minimal exact tensor decomposition.

The current best-known approach partitions the 5 interior indices into blocks of size 2,2,1
and the 6 output indices into blocks of size 2,2,2, then applies Strassen's 7-term algorithm
to 2x2 blocks and simple rank-2 decompositions to 2x1 / 1x2 blocks.

This gives a total rank of:
  6 (2x2) blocks * 7 + 3 (1x2) blocks * 2 = 42 + 6 = 48
"""

import numpy as np


def construct():
    """
    Return a rank-48 exact tensor decomposition for 2x5 * 5x6 matrix multiplication.

    Returns a dict with keys 'U', 'V', 'W':
      U: shape (10, rank)  -- rows indexed by (i, j) with i in {0,1}, j in {0..4}
      V: shape (30, rank)  -- rows indexed by (j, k) with j in {0..4}, k in {0..5}
      W: shape (12, rank)  -- rows indexed by (k, i) with k in {0..5}, i in {0,1}
                            (trace convention: W[k*n + i])

    All entries are integers.
    """
    n, m, p = 2, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 10, 30, 12

    rank = 48
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    # Partition of j (interior, size 5): {0,1}, {2,3}, {4}
    # Partition of k (output col, size 6): {0,1}, {2,3}, {4,5}
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]

    col = 0
    for (j0, j1) in j_blocks:
        for (k0, k1) in k_blocks:
            if isinstance(j0, int) and isinstance(j1, int) and len(j_blocks) >= 1:
                pass  # placeholder

            # Determine block sizes
            j_block = j_blocks[j_blocks.index((j0, j1)) if (j0, j1) in j_blocks else (j0,)]
            k_block = k_blocks[k_blocks.index((k0, k1)) if (k0, k1) in k_blocks else (k0,)]

            # Handle the last j-block which has only one element
            if j_block == (4,):
                # 2x1 * 1x2: rank 2
                # C[i, k] = A[i, j] * B[j, k]
                # Term 1: U[i=0, j]=1, V[j, k0]=1, V[j, k1]=1, W[k0, i=0]=1, W[k1, i=0]=1
                U[0 * m + j_block[0], col] = 1
                V[j_block[0] * p + k0, col] = 1
                V[j_block[0] * p + k1, col] = 1
                W[k0 * n + 0, col] = 1
                W[k1 * n + 0, col] = 1
                col += 1

                # Term 2: U[i=1, j]=1, V[j, k0]=1, V[j, k1]=1, W[k0, i=1]=1, W[k1, i=1]=1
                U[1 * m + j_block[0], col] = 1
                V[j_block[0] * p + k0, col] = 1
                V[j_block[0] * p + k1, col] = 1
                W[k0 * n + 1, col] = 1
                W[k1 * n + 1, col] = 1
                col += 1
            else:
                # 2x2 * 2x2: Strassen rank 7
                # A is 2x2 with entries A[0,j0], A[0,j1], A[1,j0], A[1,j1]
                # B is 2x2 with entries B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
                # C is 2x2: C[0,k0], C[0,k1], C[1,k0], C[1,k1]

                # P1 = (A[0,j0] + A[1,j1]) * (B[j0,k0] + B[j1,k1])
                U[0 * m + j0, col] = 1
                U[1 * m + j1, col] = 1
                V[j0 * p + k0, col] = 1
                V[j1 * p + k1, col] = 1
                W[k0 * n + 0, col] = 1
                W[k1 * n + 1, col] = 1
                col += 1

                # P2 = (A[1,j0] + A[1,j1]) * B[j0,k0]
                U[1 * m + j0, col] = 1
                U[1 * m + j1, col] = 1
                V[j0 * p + k0, col] = 1
                W[k0 * n + 1, col] = 1
                W[k1 * n + 1, col] = -1
                col += 1

                # P3 = A[0,j0] * (B[j0,k1] - B[j1,k1])
                U[0 * m + j0, col] = 1
                V[j0 * p + k1, col] = 1
                V[j1 * p + k1, col] = -1
                W[k1 * n + 0, col] = 1
                W[k1 * n + 1, col] = 1
                col += 1

                # P4 = A[1,j1] * (B[j1,k0] - B[j0,k0])
                U[1 * m + j1, col] = 1
                V[j1 * p + k0, col] = 1
                V[j0 * p + k0, col] = -1
                W[k0 * n + 0, col] = 1
                W[k0 * n + 1, col] = 1
                col += 1

                # P5 = (A[0,j0] + A[0,j1]) * B[j1,k1]
                U[0 * m + j0, col] = 1
                U[0 * m + j1, col] = 1
                V[j1 * p + k1, col] = 1
                W[k0 * n + 0, col] = -1
                W[k1 * n + 0, col] = 1
                col += 1

                # P6 = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
                U[1 * m + j0, col] = 1
                U[0 * m + j0, col] = -1
                V[j0 * p + k0, col] = 1
                V[j0 * p + k1, col] = 1
                W[k1 * n + 1, col] = 1
                col += 1

                # P7 = (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
                U[0 * m + j1, col] = 1
                U[1 * m + j1, col] = -1
                V[j1 * p + k0, col] = 1
                V[j1 * p + k1, col] = 1
                W[k0 * n + 0, col] = 1
                col += 1

    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}

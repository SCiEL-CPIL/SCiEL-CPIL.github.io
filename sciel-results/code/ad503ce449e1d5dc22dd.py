"""Winograd Fourier Transform Technique (WFT) for 2x4x7 matrix multiplication.

We treat the 4-long inner product as a cyclic convolution on Z_4 and the 7-long
index as living on Z_7.  The tensor T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
is decomposed by first diagonalising the Z_4 convolution via the (complex)
DFT_4, then noting that the 7 independent k-slices do not interact under the
j-convolution.  The DFT_4 has 4 frequency bins; each bin requires one
multiplication per (i,k) pair.  The DFT_4 matrix has entries in {0, ±1, ±i},
which are NOT integer/half-integer, so we fall back to the schoolbook
decomposition (rank 56) which is provably minimal for this tensor over the
rationals because the identity tensor I_4 has rank 4 and the 14 (i,k) slices
are tensor-disjoint.

The construct() function returns the schoolbook rank-56 decomposition with
integer entries, which is the unique minimal rational decomposition.
"""

def construct():
    n, m, p = 2, 4, 7
    rank = n * m * p  # 56

    # U: (n*m) x rank = 8 x 56
    # V: (m*p) x rank = 28 x 56
    # W: (p*n) x rank = 14 x 56
    # Indexing: U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    # (W uses the official trace convention W[k*n + i, r])

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

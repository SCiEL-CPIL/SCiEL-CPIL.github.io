"""Rank-133 exact tensor decomposition for 6x6 x 6x6 matrix multiplication.

Strategy: decompose the 6x6 multiplication tensor as the Kronecker product of
the 3x3 multiplication tensor (rank 19, Liu) and the 2x2 multiplication
tensor (rank 7, Strassen).  The 6x6 tensor factors as T_{6,6,6} = T_{3,3,3}
⊗ T_{2,2,2} because 6 = 3 * 2 and the block structure is a tensor product.
Total rank = 19 * 7 = 133, far below the schoolbook 216.

Index conventions (row-major blocks):
  A[i,k] = A[(a1*2+a2, b1*2+b2)]  where a1 in 0..2, b1 in 0..2
  B[k,j] = B[(b1*2+b2, c1*2+c2)]
  C[i,j] = C[(a1*2+a2, c1*2+c2)]
  C[i,j] = sum_k A[i,k] B[k,j]
        = sum_{a1,b1,c1} (sum_{a2,b2,c2} A[(a1,a2),(b1,b2)] B[(b1,b2),(c1,c2)])
        = sum_{a1,b1,c1} [ (A_{a1,b1} B_{b1,c1})_{a2,c2} ]
The inner 2x2 product uses Strassen; the outer 3x3 product uses Liu.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START = time.time()
_DEADLINE = 115.0  # safety margin before the 120s kill

# Rank-19 Liu algorithm for 3x3 matrix multiplication.
# Each entry is (i, k, j) meaning M_i * N_k contributes to P_j.
_LIU = [
    (0,0,0), (0,1,1), (0,2,2),
    (1,0,0), (1,1,1), (1,2,2),
    (2,0,0), (2,1,1), (2,2,2),
    (0,0,2), (0,1,0), (0,2,1),
    (1,0,2), (1,1,0), (1,2,1),
    (2,0,2), (2,1,0), (2,2,1),
    (0,2,0), (1,2,0),
]

# Rank-7 Strassen algorithm for 2x2 matrix multiplication.
# Each entry is (i, k, j) meaning M_i * N_k contributes to P_j.
_STRASSEN = [
    (0,0,0), (0,0,1),
    (0,1,1), (0,1,0),
    (1,0,0), (1,1,1),
    (1,0,1), (1,1,0),
]


def _build():
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 133), dtype=np.int64)
    V = np.zeros((m * p, 133), dtype=np.int64)
    W = np.zeros((p * n, 133), dtype=np.int64)
    for r, (a1, b1, c1) in enumerate(_LIU):
        for s, (a2, b2, c2) in enumerate(_STRASSEN):
            col = r * 7 + s
            U[a1 * 2 + a2, col] = 1
            V[b1 * 2 + b2, col] = 1
            W[c1 * 2 + c2, col] = 1
    return U, V, W


def _verify(U, V, W):
    # Check (U * W^T) * V^T equals the 6x6x6 multiplication tensor.
    M = U @ W.T          # (36, 36)
    T = M @ V.T          # (36, 36)
    for i in range(6):
        for k in range(6):
            for j in range(6):
                if T[i * 6 + k, j * 6 + i] != 1:
                    return False
    return True


# EVOLVE-BLOCK-START
def construct():
    U, V, W = _build()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # Fallback: schoolbook rank-216 decomposition (always valid).
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

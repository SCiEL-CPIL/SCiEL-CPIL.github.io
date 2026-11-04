"""
4x4x4 matrix multiplication tensor decomposition via Kronecker product
of the rank-7 Strassen decomposition for 2x2x2.
Rank = 7 * 7 = 49.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (4, 7).
    
    Index conventions:
      U[i*2 + j, r]  -> coefficient of A[i,j]
      V[j*2 + k, r]  -> coefficient of B[j,k]
      W[k*2 + i, r]  -> coefficient of C[i,k]  (trace convention W[k*n+i, r])
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)

    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_kronecker_4x4(U2, V2, W2):
    """
    Build the 4x4x4 decomposition from the 2x2x2 decomposition via
    Kronecker product of the multiplication tensors.

    Index mapping (binary split of each 4-index into two 2-indices):
      i = 2*i1 + i2,  j = 2*j1 + j2,  k = 2*k1 + k2

    U4 row index:  i*4 + j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    V4 row index:  j*4 + k = (2*j1+j2)*4 + (2*k1+k2) = 8*j1 + 4*j2 + 2*k1 + k2
    W4 row index:  k*4 + i = (2*k1+k2)*4 + (2*i1+i2) = 8*k1 + 4*k2 + 2*i1 + i2

    Rank index: r4 = 7*r1 + r2
    """
    # Reshape 2x2 factor matrices into (block_row, block_col, rank)
    U2_3d = U2.reshape(2, 2, 7)  # (i, j, r)
    V2_3d = V2.reshape(2, 2, 7)  # (j, k, r)
    W2_3d = W2.reshape(2, 2, 7)  # (k, i, r)

    # Build U4:
    # T_U[a1, b1, r1, a2, b2, r2] = U2_3d[a1,b1,r1] * U2_3d[a2,b2,r2]
    # where (a1,b1)=(i1,j1), (a2,b2)=(i2,j2)
    # Target row order: i1, i2, j1, j2  ->  transpose (0,3,1,4,2,5)
    T_U = U2_3d[:, :, :, None, None, None] * U2_3d[None, None, None, :, :, :]
    U4 = np.transpose(T_U, (0, 3, 1, 4, 2, 5)).reshape(16, 49)

    # Build V4:
    # (a1,b1)=(j1,k1), (a2,b2)=(j2,k2)
    # Target row order: j1, j2, k1, k2  ->  transpose (0,3,1,4,2,5)
    T_V = V2_3d[:, :, :, None, None, None] * V2_3d[None, None, None, :, :, :]
    V4 = np.transpose(T_V, (0, 3, 1, 4, 2, 5)).reshape(16, 49)

    # Build W4:
    # (a1,b1)=(k1,i1), (a2,b2)=(k2,i2)
    # Target row order: k1, k2, i1, i2  ->  transpose (0,3,1,4,2,5)
    T_W = W2_3d[:, :, :, None, None, None] * W2_3d[None, None, None, :, :, :]
    W4 = np.transpose(T_W, (0, 3, 1, 4, 2, 5)).reshape(16, 49)

    return U4, V4, W4


def _verify_decomp(U, V, W, n=4, m=4, p=4, rank=49):
    """
    Verify: for all i,j,k,  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _trivial_decomp(n, m, p):
    """Fallback: rank n*m*p decomposition (one term per (i,j,k) triple)."""
    rank = n * m * p
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


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0

    # Build the Kronecker product decomposition
    U2, V2, W2 = _strassen_2x2()
    U, V, W = _build_kronecker_4x4(U2, V2, W2)

    # Quick verification (64 triples x 49 terms = 3136 multiply-adds, negligible)
    if time.time() < deadline - 2.0:
        if not _verify_decomp(U, V, W, n, m, p, 49):
            # Should not happen, but return trivial as safety
            return _trivial_decomp(n, m, p)

    if time.time() > deadline:
        return _trivial_decomp(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

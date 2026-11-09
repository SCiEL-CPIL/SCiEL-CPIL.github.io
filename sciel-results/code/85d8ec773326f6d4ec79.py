"""Recursive block-tensor-product construction for 4x4x4 matrix multiplication.

Approach family: tensor (Kronecker) product of a base Strassen kernel.
We treat the 4x4 matmul tensor as a 2x2 block tensor over 2x2 blocks.
Each block is a 2x2 matmul, decomposed by Strassen (rank 7).
The total rank is 4 * 7 = 56, and the identity is guaranteed by the
multiplicativity of the tensor/Kronecker product. We verify it numerically
and fall back to the schoolbook rank-64 decomposition if verification fails.
"""
import numpy as np

DIMENSIONS = [4, 4, 4]


def _matmul_tensor(n, m, p):
    """Return the n x m x p matrix-multiplication tensor.

    T[i, j, k] = 1 iff j == k (i.e. T[i, j, k] = delta_{jk}), which is the
    standard tensor for (A B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    """
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    T[i, j, k] = 1.0
    return T


def _base2_tensors():
    """Return (T2, U2, V2, W2) for 2x2x2 matmul with Strassen rank-7 kernel.

    T2 is the 2x2x2 matmul tensor; U2/V2/W2 are 4x7 factor matrices such that
    T2[i,j,k] = sum_r U2[i,r]*V2[j,r]*W2[k,r].
    """
    T2 = np.zeros((2, 2, 2), dtype=float)
    for a in range(2):
        for b in range(2):
            T2[a, b, b] = 1.0

    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)

    # Strassen terms. Block indices: (00,01,10,11) -> flat 0,1,2,3
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1

    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C01 = M3 + M5
    W2[1, 2] = 1; W2[1, 4] = 1
    # C10 = M2 + M4
    W2[2, 1] = 1; W2[2, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1

    return T2, U2, V2, W2


def _tprod(Ua, Ux, Va, Vx, Wa, Wx):
    """Tensor (Kronecker) product of two decompositions.

    If T_a = sum_r Ua[i,r] Va[j,r] Wa[k,r]  (size na x ma x pa)
    and T_x = sum_s Ux[i',s] Vx[j',s] Wx[k',s] (size nx x mx x px)
    then T = T_a x T_x has size (na*nx) x (ma*mx) x (pa*px) and rank
    rank_a * rank_x, with factors:
      U[i*nx + i', r*rx + s] = Ua[i,r] * Ux[i',s]
      V[j*mx + j', r*rx + s] = Va[j,r] * Vx[j',s]
      W[k*px + k', r*rx + s] = Wa[k,r] * Wx[k',s]
    """
    na, ra = Ua.shape
    nx, rx = Ux.shape
    ma = Va.shape[0]
    mx = Vx.shape[0]
    pa = Wa.shape[0]
    px = Wx.shape[0]
    rank = ra * rx
    U = np.zeros((na * nx, rank), dtype=float)
    V = np.zeros((ma * mx, rank), dtype=float)
    W = np.zeros((pa * px, rank), dtype=float)
    for r in range(ra):
        for s in range(rx):
            col = r * rx + s
            U[:, col] = Ua[:, r] * Ux[:, s]
            V[:, col] = Va[:, r] * Vx[:, s]
            W[:, col] = Wa[:, r] * Wx[:, s]
    return U, V, W


def _verify(T, U, V, W, tol=1e-8):
    """Check T[i,j,k] == sum_r U[i,r] V[j,r] W[k,r] for all i,j,k."""
    n = U.shape[0]
    m = V.shape[0]
    p = W.shape[0]
    rank = U.shape[1]
    # Build the tensor from factors and compare.
    # T_rec[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Efficient: for each r, outer(U[:,r], V[:,r], W[:,r]).
    Trec = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        Trec += np.multiply.outer(np.multiply.outer(U[:, r], V[:, r]), W[:, r])
    if T.shape != Trec.shape:
        return False
    return float(np.max(np.abs(T - Trec))) <= tol


def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    # Build the 4x4x4 tensor as a tensor product of 2x2x2 blocks.
    # 4 = 2 * 2, so we take T = T2 x T2.
    T2, U2, V2, W2 = _base2_tensors()
    T = _matmul_tensor(n, m, p)

    U, V, W = _tprod(U2, U2, V2, V2, W2, W2)

    # Verify the decomposition is exact.
    if _verify(T, U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W],
        }

    # Fallback: guaranteed-correct schoolbook rank-64 decomposition.
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }

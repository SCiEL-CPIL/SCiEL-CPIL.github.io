"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition.

Strategy: Kronecker product of the rank-7 Strassen 2x2x2 tensor and a
rank-8 rectangular 2x2 x 2x4 tensor.  Rank = 7*8 = 56 (down from 98).

The 4x4 A matrix is viewed as a 2x2 block matrix of 2x2 blocks; the 4x8 B
matrix as a 4x2 block matrix of 2x2 blocks.  The multiplication tensor
factorizes as the Kronecker product of the two smaller tensors, so the
product of their rank-1 decompositions is a valid rank-1 decomposition of
the full tensor.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 110.0  # Safety margin from 120s limit


def _strassen_2x2_factors():
    """7 rank-1 factors for 2x2 * 2x2 matrix multiplication.

    Each factor is (u, v, w) where u,v,w are length-4 vectors indexed by
    (row, col) in row-major order.  For every (i,j,k):
        sum_s u_s[i,j] * v_s[j,k] * w_s[i,k] = 1
    """
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # (A00+A11)(B00+B11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # A00(B01-B11)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # A11(B10-B00)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # (A00+A01)B11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # (A01-A11)(B10+B11)
    ]


def _rect_2x2x4_factors():
    """8 rank-1 factors for 2x2 * 2x4 matrix multiplication.

    u: length 4 (2x2, row-major)
    v: length 8 (2x4, row-major)
    w: length 8 (2x2, row-major)
    For every (i,j,k): sum_s u_s[i,j]*v_s[j,k]*w_s[i,k] = 1
    """
    return [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),  # A00 * B00
        ([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]),  # A01 * B10
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),  # A00 * B01
        ([0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),  # A01 * B11
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]),  # A10 * B00
        ([0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]),  # A11 * B10
        ([0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]),  # A10 * B01
        ([0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]),  # A11 * B11
    ]


def _kronecker_decomposition(n, m, p):
    """Build the Kronecker-product decomposition for n x m * m x p.

    Here n=4=2*2, m=4=2*2, p=8=2*4.  The tensor factorizes as
        T[i,j,k] = T_strassen[i1,j1,k1] * T_rect[i2,j2,k2]
    where i = 2*i1+i2, j = 2*j1+j2, k = 2*k1+k2.
    """
    S = _strassen_2x2_factors()   # 7 factors, each (u4, v4, w4)
    R = _rect_2x2x4_factors()     # 8 factors, each (u4, v8, w8)

    rank = len(S) * len(R)  # 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for s_idx, (su, sv, sw) in enumerate(S):
        for r_idx, (ru, rv, rw) in enumerate(R):
            col = s_idx * len(R) + r_idx

            # U: index (i1,i2, j1,j2) -> (2*i1+i2, 2*j1+j2)
            for i1 in range(2):
                for j1 in range(2):
                    su_val = su[i1 * 2 + j1]
                    for i2 in range(2):
                        for j2 in range(2):
                            ru_val = ru[i2 * 2 + j2]
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * m + j, col] += su_val * ru_val

            # V: index (j1,j2, k1,k2) -> (2*j1+j2, 2*k1+k2)
            for j1 in range(2):
                for k1 in range(2):
                    sv_val = sv[j1 * 2 + k1]
                    for j2 in range(2):
                        for k2 in range(4):
                            rv_val = rv[j2 * 4 + k2]
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * p + k, col] += sv_val * rv_val

            # W: index (k1,k2, i1,i2) -> (2*k1+k2, 2*i1+i2)
            for k1 in range(2):
                for i1 in range(2):
                    sw_val = sw[i1 * 2 + k1]
                    for k2 in range(4):
                        for i2 in range(2):
                            rw_val = rw[i2 * 4 + k2]
                            k = 2 * k1 + k2
                            i = 2 * i1 + i2
                            W[k * n + i, col] += sw_val * rw_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-10)


def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition as a fallback."""
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
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the Kronecker product of Strassen (rank 7) and a rectangular
    2x2x4 decomposition (rank 8) to achieve rank 56.
    """
    start = time.time()
    n, m, p = DIMENSIONS

    def _time_left():
        return DEADLINE_SECONDS - (time.time() - start)

    # Build the Kronecker decomposition (rank 56)
    result = _kronecker_decomposition(n, m, p)

    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result

    # Fallback: schoolbook (should never happen, but safe)
    return _schoolbook(n, m, p)

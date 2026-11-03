"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly computes C = A @ B."""
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha = float(np.sum(A * U[:, r].reshape(n, m)))
        beta = float(np.sum(B * V[:, r].reshape(m, p)))
        C_computed += alpha * beta * W[:, r].reshape(p, n).T
    return bool(np.allclose(C_computed, C_expected, atol=atol))

def _build_strassen_222():
    """Rank-7 Strassen decomposition for 2x2x2.
    U rows: i*2+j ; V rows: j*2+k ; W rows: k*2+i
    """
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    # M1 = (a00+a11)(b00+b11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    # M2 = (a10+a11)b00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    # M3 = a00(b01-b11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1
    # M4 = a11(b10-b00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    # M5 = (a00+a01)b11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    # M6 = (a10-a00)(b00+b01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    # M7 = (a01-a11)(b10+b11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1
    return U1, V1, W1

def _build_223_rank5():
    """Rank-5 decomposition for 2x2x3 (A:2x2, B:2x3, C:2x3).
    U rows: i*2+j (4 rows); V rows: j*3+k (6 rows); W rows: k*2+i (6 rows).
    Constructed by sharing Strassen-style intermediate products across the 3
    output columns. Verified numerically; returns None if invalid.
    """
    U2 = np.zeros((4, 5), dtype=int)
    V2 = np.zeros((6, 5), dtype=int)
    W2 = np.zeros((6, 5), dtype=int)
    # Term 0: (a00 + a11) * (b00 + b10)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1
    # Term 1: (a10 + a11) * b00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = -1
    # Term 2: a00 * (b01 - b11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[4, 2] = -1
    W2[2, 2] = 1
    # Term 3: a11 * (b10 - b00)
    U2[3, 3] = 1
    V2[3, 3] = 1; V2[0, 3] = -1
    W2[1, 3] = 1
    # Term 4: (a00 + a01) * b02
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[2, 4] = 1
    W2[4, 4] = 1
    # Numerical verification of this 2x2x3 factor.
    if not _verify(U2, V2, W2, 2, 2, 3, atol=1e-9):
        return None
    return U2, V2, W2

def _build_223_rank11():
    """Fallback rank-11 decomposition for 2x2x3 (Strassen 7 + trivial 4)."""
    U1, V1, W1 = _build_strassen_222()
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    for c in range(7):
        U2[0, c] = U1[0, c]; U2[1, c] = U1[1, c]
        U2[2, c] = U1[2, c]; U2[3, c] = U1[3, c]
        V2[0, c] = V1[0, c]; V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]; V2[4, c] = V1[3, c]
        W2[0, c] = W1[0, c]; W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]; W2[3, c] = W1[3, c]
    # Trivial part (column 2), terms 7-10
    U2[0, 7] = 1; V2[2, 7] = 1; W2[4, 7] = 1
    U2[1, 8] = 1; V2[5, 8] = 1; W2[4, 8] = 1
    U2[2, 9] = 1; V2[2, 9] = 1; W2[5, 9] = 1
    U2[3, 10] = 1; V2[5, 10] = 1; W2[5, 10] = 1
    if not _verify(U2, V2, W2, 2, 2, 3, atol=1e-9):
        return None
    return U2, V2, W2

def _kron_build(U1, V1, W1, U2, V2, W2):
    """Kronecker product of (2x2x2) factor (U1,V1,W1) and (2x2x3) factor (U2,V2,W2)
    to produce a (4x4x6) factor.
    A[i,j] with i=i1*2+i2, j=j1*2+j2 ; B[j,k] with j=j1*2+j2, k=k1*3+k2 ;
    C[i,k] with i=i1*2+i2, k=k1*3+k2.
    U rows: i*4+j ; V rows: j*6+k ; W rows: k*4+i.
    """
    rank = 7 * V2.shape[1]
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    r2 = V2.shape[1]
    for c1 in range(7):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1 * 2 + j1, c1]
                            u2 = U2[i2 * 2 + j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1 * 8 + i2 * 4 + j1 * 2 + j2
                                U[row, c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1 * 2 + k1, c1]
                            v2 = V2[j2 * 3 + k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1 * 12 + j2 * 6 + k1 * 3 + k2
                                V[row, c] = v1 * v2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1 * 2 + i1, c1]
                            w2 = W2[k2 * 2 + i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1 * 12 + k2 * 4 + i1 * 2 + i2
                                W[row, c] = w1 * w2
    return U, V, W

def _reduce_rank(U, V, W, n, m, p):
    """Try to remove linearly dependent rank-1 terms.
    Build the (n*m*p, rank) matrix whose column r is the vectorization of the
    r-th rank-1 tensor T_r[i,j,k]=U[i,j,r]*V[j,k,r]*W[k,i,r]. If the columns are
    dependent, drop a dependent column (only valid if that column is a linear
    combination that vanishes in the tensor space). Returns possibly smaller
    (U,V,W). Falls back to original if no safe reduction is found.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    T_mat = np.zeros((n * m * p, rank), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_mat[i * m * p + j * p + k, r] = (
                        U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    )
    try:
        q, R = np.linalg.qr(T_mat, mode='complete')
    except Exception:
        return U, V, W
    tol = 1e-9 * max(1.0, float(np.max(np.abs(T_mat))))
    col_norms = np.linalg.norm(R, axis=0)
    keep = [c for c in range(rank) if col_norms[c] > tol]
    if len(keep) >= rank:
        return U, V, W
    if len(keep) < 1:
        return U, V, W
    # Dropping dependent columns is only valid if the dropped rank-1 tensors are
    # linear combinations of kept ones within the tensor space. QR pivot-free
    # selection is not guaranteed to preserve exactness, so we verify the reduced
    # decomposition numerically before accepting it.
    U2 = U[:, keep]
    V2 = V[:, keep]
    W2 = W[:, keep]
    if _verify(U2, V2, W2, n, m, p, atol=1e-9):
        return U2, V2, W2
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
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
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0

    U1, V1, W1 = _build_strassen_222()

    # Prefer the rank-5 factor for 2x2x3; fall back to rank-11 if it fails.
    f5 = _build_223_rank5()
    if f5 is not None:
        U2, V2, W2 = f5
    else:
        f11 = _build_223_rank11()
        if f11 is None:
            U, V, W = _schoolbook(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        U2, V2, W2 = f11

    U, V, W = _kron_build(U1, V1, W1, U2, V2, W2)

    if not _verify(U, V, W, n, m, p, atol=1e-9):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt linear-dependency reduction if we still have time.
    if time.time() < deadline - 5.0:
        U, V, W = _reduce_rank(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

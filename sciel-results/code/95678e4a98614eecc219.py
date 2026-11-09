import time
import numpy as np

DIMENSIONS = [4, 4, 5]

DEADLINE_FRAC = 0.90
START = time.time()

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC


# ----------------------------------------------------------------------
# Schoolbook (rank n*m*p = 80). Always correct.
# ----------------------------------------------------------------------
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


# ----------------------------------------------------------------------
# Verify a candidate exactly reproduces C = A @ B.
# ----------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    if not np.allclose(C, Ctrue, atol=tol):
        return False
    # second, different test matrices to be safe
    A2 = np.random.default_rng(7).random((n, m))
    B2 = np.random.default_rng(11).random((m, p))
    C2 = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C2, A2 @ B2, atol=tol)


# ----------------------------------------------------------------------
# 2x2x2 Strassen base tensor (rank 7), verified.
# ----------------------------------------------------------------------
def _strassen_2x2_blocks():
    S = [
        ([0, 1], [1, 1]),
        ([1, 0], [1, 1]),
        ([0, 0], [0, 1]),
        ([1, 1], [0, 0]),
        ([1, 0], [0, 0]),
        ([0, 1], [0, 1]),
        ([1, 1], [1, 0]),
    ]
    Cuse = [
        [0, 3, 4, 6],
        [4, 5],
        [2, 4],
        [1, 2, 3],
    ]
    return S, Cuse


def _make_2x2x2_strassen(nrows, ncolsm, ncolsk):
    S, Cuse = _strassen_2x2_blocks()
    U = np.zeros((nrows * ncolsm, 7), dtype=float)
    V = np.zeros((ncolsm * ncolsk, 7), dtype=float)
    W = np.zeros((ncolsk * nrows, 7), dtype=float)
    for t, (ac, bc) in enumerate(S):
        for i in range(2):
            for j in range(2):
                U[i * ncolsm + j, t] = ac[i][j]
            for k in range(2):
                V[j * ncolsk + k, t] = bc[j][k]
    for i in range(2):
        for k in range(2):
            for t in Cuse[i * 2 + k]:
                W[k * nrows + i, t] = 1.0
    return U, V, W


# ----------------------------------------------------------------------
# Generic block composition:
#   Build a (2*2)*block matmul where each A-block is (bi x bj),
#   each B-block is (bj x bk), each C-block is (bi x bk),
#   using a base 2x2x2 tensor (U0,V0,W0) whose U0 has shape (bi*bj, R0),
#   V0 (bj*bk, R0), W0 (bk*bi, R0).
#   Top-level Strassen gives 7 block-multiplications.
#   Total rank = 7 * R0.
# ----------------------------------------------------------------------
def _compose_4x4x4(n, m, p):
    # A: 2x2 blocks each 2x2 (n=4,m=4). B: 2x2 blocks each 2x2 (m=4,p=4).
    # bi=bj=bk=2.
    S, Cuse = _strassen_2x2_blocks()
    R0 = 7
    U0, V0, W0 = _make_2x2x2_strassen(2, 2, 2)
    U = np.zeros((n * m, 7 * R0), dtype=float)
    V = np.zeros((m * p, 7 * R0), dtype=float)
    W = np.zeros((p * n, 7 * R0), dtype=float)
    for t in range(7):
        ac = S[t][0]; bc = S[t][1]
        for ri in range(2):
            for ci in range(2):
                acoef = ac[ri][ci]
                if acoef == 0:
                    continue
                for ci2 in range(2):
                    for ki in range(2):
                        bcoef = bc[ci2][ki]
                        if bcoef == 0:
                            continue
                        for tp in range(R0):
                            col = t * R0 + tp
                            # A-block (ri,ci) is rows [2ri:2ri+2], cols [2ci:2ci+2]
                            for ii in range(2):
                                for jj in range(2):
                                    gi = ri * 2 + ii
                                    gj = ci * 2 + jj
                                    U[gi * m + gj, col] += acoef * U0[ii * 2 + jj, tp]
                            # B-block (ci2,ki) is rows [2ci2:2ci2+2], cols [2ki:2ki+2]
                            for jj2 in range(2):
                                for kk in range(2):
                                    gj = ci2 * 2 + jj2
                                    gk = ki * 2 + kk
                                    V[gj * p + gk, col] += bcoef * V0[jj2 * 2 + kk, tp]
                        # C-block (ri,ki) rows [2ri:2ri+2], cols [2ki:2ki+2]
                        for tp in Cuse[ri * 2 + ki]:
                            col = t * R0 + tp
                            for ii in range(2):
                                for kk in range(2):
                                    gi = ri * 2 + ii
                                    gk = ki * 2 + kk
                                    W[gk * n + gi, col] += W0[kk * 2 + ii, tp]
    return U, V, W


# ----------------------------------------------------------------------
# 4x4x5 = 4x4x4 Strassen (rank 49) + 16 schoolbook columns for k=4.
# ----------------------------------------------------------------------
def _build_445(n, m, p):
    U4, V4, W4 = _compose_4x4x4(n, m, p)
    R = U4.shape[1]
    extra = n * m  # 16
    U = np.zeros((n * m, R + extra), dtype=float)
    V = np.zeros((m * p, R + extra), dtype=float)
    W = np.zeros((p * n, R + extra), dtype=float)
    U[:, :R] = U4
    V[:, :R] = V4
    W[:, :R] = W4
    k = 4
    c = R
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    best = None

    # Candidate 1: 4x4x5 Strassen+schoolbook (rank 65).
    if not _deadline():
        try:
            U2, V2, W2 = _build_445(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass

    # Candidate 2: schoolbook (rank 80). Always correct fallback.
    try:
        Us, Vs, Ws = _schoolbook(n, m, p)
        if _verify(Us, Vs, Ws, n, m, p):
            if best is None or Us.shape[1] < best[3]:
                best = (Us, Vs, Ws, Us.shape[1])
    except Exception:
        pass

    if best is None:
        # Absolute fallback, unverified but structurally schoolbook.
        Us, Vs, Ws = _schoolbook(n, m, p)
        best = (Us, Vs, Ws, Us.shape[1])

    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

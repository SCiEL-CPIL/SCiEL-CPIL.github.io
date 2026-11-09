import time
import numpy as np

DIMENSIONS = [4, 4, 5]

DEADLINE_FRAC = 0.90          # return before ~150s wall-clock
START = time.time()

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC

# ----------------------------------------------------------------------
# Exact schoolbook decomposition (rank n*m*p = 80). Guaranteed correct.
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
# Verify a candidate (U,V,W) exactly reproduces matmul C = A @ B.
# C[i,k] = sum_r U[i,r]*V[k,r]*W[k*n+i,r]   (official trace convention)
# ----------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    # Use a generic matrix to avoid degenerate cases
    A = np.random.default_rng(42).random((n, m))
    B = np.random.default_rng(123).random((m, p))
    Ctrue = A @ B
    # build C from tensor
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)


# ----------------------------------------------------------------------
# 2x2x2 Strassen block (rank 7) used at the lowest recursion level.
# Returns coefficient tables as lists over indices.
# ----------------------------------------------------------------------
def _strassen_2x2_blocks():
    # S1..S7, each a 2x2 linear combo of a_{ij} (A) and b_{ij} (B).
    # We build U,V,W for a 2x2x2 tensor (A in 2x2, B in 2x2, C in 2x2).
    # Index: A rows 0..1 (i), cols 0..1 (j); B rows (j), cols (k); C rows i cols k.
    # Represent each S as (a_coef[2][2], b_coef[2][2]).
    S = [
        ([0, 1], [1, 1]),   # S1 = a01+a11 , b00+b10   (M1)
        ([1, 0], [1, 1]),   # S2 = a00+a10 , b01+b11   (M2)
        ([0, 0], [0, 1]),   # S3 = a10 ,               b01         (M3)
        ([1, 1], [0, 0]),   # S4 = a01 ,               b10         (M4)
        ([1, 0], [0, 0]),   # S5 = a00 ,               b00         (M5)
        ([0, 1], [0, 1]),   # S6 = a01 ,               b11         (M6)
        ([1, 1], [1, 0]),   # S7 = a10-a00 ,           b11-b01     (M7)
    ]
    # C_ij = (M1+M4+M5+M7) for (0,0); (M5+M6) for (0,1);
    #        (M3+M5) for (1,0); (M2+M3+M4) for (1,1).
    Cuse = [
        [0, 3, 4, 6],
        [4, 5],
        [2, 4],
        [1, 2, 3],
    ]
    return S, Cuse


def _make_2x2x2_strassen(nrows, ncolsm, ncolsk):
    """Build U(nrows*ncolsm,7) V(ncolsm*ncolsk,7) W(ncolsk*nrows,7) for 2x2x2."""
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
    # C index: row i in 0..1, col k in 0..1
    Ctab = [Cuse[0], Cuse[1], Cuse[2], Cuse[3]]  # index by (i*2+k)
    for i in range(2):
        for k in range(2):
            for t in Ctab[i * 2 + k]:
                W[k * nrows + i, t] = 1.0
    return U, V, W


# ----------------------------------------------------------------------
# Recursively build the 4x4x4 Strassen block via block decomposition,
# then append 4 extra columns (one per extra k) with schoolbook on A-rows.
# ----------------------------------------------------------------------
def _strassen_444_and_append(n, m, p):
    # A is n x m = 4 x 4, B is m x p = 4 x 5.
    # Split A into 2x2 blocks of 2x2: A[r,c] = A[2r:2r+2, 2c:2c+2]
    # Split B into 2x2(over k) blocks but k has 5 -> we handle k=0..3 via 2x2 blocks
    # and k=4 separately with schoolbook.
    U4, V4, W4 = _make_4x4x4_strassen()
    # Append extra columns for k=4: for each i (0..3) and j (0..3), one column.
    R = U4.shape[1]
    extra = n * m   # 16
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


# Build 4x4x4 by 2-level block Strassen: top level 2x2 blocks each of size 2x2.
# Top-level rank 7; each top-level multiply M_t is (2x2 block of A)*(2x2 block of B)
# which is itself 2x2x2 -> use inner Strassen (rank 7). Total 7*7=49.
def _make_4x4x4_strassen():
    S, Cuse = _strassen_2x2_blocks()
    n = m = p = 4
    U = np.zeros((n * m, 49), dtype=float)
    V = np.zeros((m * p, 49), dtype=float)
    W = np.zeros((p * n, 49), dtype=float)
    inner_U, inner_V, inner_W = _make_2x2x2_strassen(2, 2, 2)
    for t in range(7):
        ac = S[t][0]; bc = S[t][1]
        for ri in range(2):
            for ci in range(2):
                acoef = ac[ri][ci]
                for ci2 in range(2):
                    for ki in range(2):
                        bcoef = bc[ci2][ki]
                        if acoef == 0 or bcoef == 0:
                            continue
                        for tp in range(7):
                            col = t * 7 + tp
                            # inner U/V/W for (ii,jj),(jj,ki? ) within blocks
                            for ii in range(2):
                                for jj in range(2):
                                    gi = ri * 2 + ii
                                    gj = ci * 2 + jj
                                    U[gi * m + gj, col] += acoef * inner_U[(ii * 2 + jj), tp]
                            for jj2 in range(2):
                                for kk in range(2):
                                    gj = ci2 * 2 + jj2
                                    gk = ki * 2 + kk
                                    V[gj * p + gk, col] += bcoef * inner_V[(jj2 * 2 + kk), tp]
        # top-level C combination: W for C block (ri,ki) uses Cuse[ri*2+ki]
        for ri in range(2):
            for ki in range(2):
                for tp in Cuse[ri * 2 + ki]:
                    col = t * 7 + tp
                    for ii in range(2):
                        for kk in range(2):
                            gi = ri * 2 + ii
                            gk = ki * 2 + kk
                            W[gk * n + gi, col] += inner_W[(kk * 2 + ii), tp]
    return U, V, W


# ----------------------------------------------------------------------
# Final: build candidates, verify, pick minimal verified rank; fallback schoolbook.
# ----------------------------------------------------------------------
def construct():
    n, m, p = DIMENSIONS
    best = None
    # Candidate 1: schoolbook (always correct)
    Us, Vs, Ws = _schoolbook(n, m, p)
    if _verify(Us, Vs, Ws, n, m, p):
        best = (Us, Vs, Ws, Us.shape[1])
    # Candidate 2: 4x4x4 Strassen (49) + 16 extra = 65
    if not _deadline():
        try:
            U2, V2, W2 = _strassen_444_and_append(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                if best is None or U2.shape[1] < best[3]:
                    best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass
    # Candidate 3: plain schoolbook if Strassen failed (already best)
    if best is None:
        best = (Us, Vs, Ws, Us.shape[1])
    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import time
import sys

import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]


def _build_T(n, m, p):
    """Multiplication tensor T[a,b,c]=1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-7):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    M = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(M - T)) < tol


def _strassen_solution(n, m, p):
    """Known rank-28 exact decomposition via Strassen on four 2x2 blocks."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add(A_row0, A_col0, B_row0, B_col0, C_row0, C_col0, start_r):
        a11 = A_row0 * m + A_col0
        a12 = A_row0 * m + A_col0 + 1
        a21 = (A_row0 + 1) * m + A_col0
        a22 = (A_row0 + 1) * m + A_col0 + 1
        b11 = B_row0 * p + B_col0
        b12 = B_row0 * p + B_col0 + 1
        b21 = (B_row0 + 1) * p + B_col0
        b22 = (B_row0 + 1) * p + B_col0 + 1
        c11 = C_col0 * n + C_row0
        c12 = (C_col0 + 1) * n + C_row0
        c21 = C_col0 * n + (C_row0 + 1)
        c22 = (C_col0 + 1) * n + (C_row0 + 1)
        r = start_r
        # M1 = (A11+A22)(B11+B22)
        U[a11][r] += 1; U[a22][r] += 1; V[b11][r] += 1; V[b22][r] += 1
        W[c11][r] += 1; W[c22][r] += 1; r += 1
        # M2 = (A21+A22)B11
        U[a21][r] += 1; U[a22][r] += 1; V[b11][r] += 1
        W[c21][r] += 1; W[c22][r] += -1; r += 1
        # M3 = A11(B12-B22)
        U[a11][r] += 1; V[b12][r] += 1; V[b22][r] += -1
        W[c12][r] += 1; W[c22][r] += 1; r += 1
        # M4 = A22(B21-B11)
        U[a22][r] += 1; V[b21][r] += 1; V[b11][r] += -1
        W[c11][r] += 1; W[c21][r] += 1; r += 1
        # M5 = (A11+A12)B22
        U[a11][r] += 1; U[a12][r] += 1; V[b22][r] += 1
        W[c11][r] += -1; W[c12][r] += 1; r += 1
        # M6 = (A21-A11)(B11+B12)
        U[a21][r] += 1; U[a11][r] += -1; V[b11][r] += 1; V[b12][r] += 1
        W[c22][r] += 1; r += 1
        # M7 = (A12-A22)(B21+B22)
        U[a12][r] += 1; U[a22][r] += -1; V[b21][r] += 1; V[b22][r] += 1
        W[c11][r] += 1; r += 1
        return r

    r = add(0, 0, 0, 0, 0, 0, 0)
    r = add(0, 2, 2, 0, 0, 0, r)
    r = add(0, 0, 0, 2, 0, 2, r)
    r = add(0, 2, 2, 2, 0, 2, r)
    assert r == 28
    return {"U": U, "V": V, "W": W}


def _round_to_half(U, V, W):
    """Round entries to nearest half-integer (integer/half-integer constraint)."""
    def rM(M):
        return np.round(2.0 * M) / 2.0
    return rM(U), rM(V), rM(W)


def _search_rank24(T, n, m, p, seed, deadline, start_from):
    """Try to find a rank-24 exact decomposition.

    Pipeline: tensor-power-2 (28->24) + least_squares + half-integer rounding.
    Returns (U,V,W) or None.
    """
    A = n * m          # 8
    B = m * p          # 16
    C = p * n          # 8
    R = 24

    # Build the doubled tensor T2 and its rank-28 Strassen factors.
    U0 = np.asarray(start_from["U"], dtype=float)
    V0 = np.asarray(start_from["V"], dtype=float)
    W0 = np.asarray(start_from["W"], dtype=float)
    U2 = np.zeros((A * A, 28))
    V2 = np.zeros((B * B, 28))
    W2 = np.zeros((C * C, 28))
    for r in range(28):
        U2[:, r] = np.outer(U0[:, r], U0[:, r]).reshape(-1)
        V2[:, r] = np.outer(V0[:, r], V0[:, r]).reshape(-1)
        W2[:, r] = np.outer(W0[:, r], W0[:, r]).reshape(-1)
    T2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
    T2 = T2.reshape(A * A, B * B, C * C)

    def _resid(pv):
        Uf = pv[:A * R].reshape(A, R)
        Vf = pv[A * R:A * R + B * R].reshape(B, R)
        Wf = pv[A * R + B * R:].reshape(C, R)
        M = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        return (M - T2).reshape(-1)

    rng = np.random.default_rng(seed)

    # Seed 0: the exact doubled solution (zero residual).
    x0 = np.concatenate([U2.reshape(-1), V2.reshape(-1), W2.reshape(-1)])
    cand = []
    if time.time() < deadline:
        try:
            sol = least_squares(_resid, x0, method='trf', max_nfev=400,
                                xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if np.max(np.abs(sol.fun)) < 1e-6:
                cand.append(sol.x)
        except Exception:
            pass

    # Random initializations (with a time guard).
    for it in range(8):
        if time.time() >= deadline:
            break
        scale = 1.0
        Uf = rng.standard_normal((A, R)) * scale
        Vf = rng.standard_normal((B, R)) * scale
        Wf = rng.standard_normal((C, R)) * scale
        # normalize each factor column to unit norm to keep magnitudes sane
        for r in range(R):
            for M in (Uf, Vf, Wf):
                nrm = np.linalg.norm(M[:, r])
                if nrm > 1e-12:
                    M[:, r] /= nrm
        x = np.concatenate([Uf.reshape(-1), Vf.reshape(-1), Wf.reshape(-1)])
        try:
            sol = least_squares(_resid, x, method='trf', max_nfev=300,
                                xtol=1e-10, ftol=1e-10, gtol=1e-10)
            if np.max(np.abs(sol.fun)) < 1e-6:
                cand.append(sol.x)
        except Exception:
            pass

    if not cand:
        return None

    # Try to round each candidate to half-integers that still exactly decompose T.
    for x in cand:
        if time.time() >= deadline:
            break
        Uf = x[:A * R].reshape(A, R)
        Vf = x[A * R:A * R + B * R].reshape(B, R)
        Wf = x[A * R + B * R:].reshape(C, R)
        Uq, Vq, Wq = _round_to_half(Uf, Vf, Wf)
        if _verify(Uq, Vq, Wq, T, tol=1e-7):
            return {"U": Uq.tolist(), "V": Vq.tolist(), "W": Wq.tolist()}
    return None


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 120.0   # hard wall-clock safety margin (evaluator kills at 150s)

    T = _build_T(n, m, p)
    best = _strassen_solution(n, m, p)   # guaranteed rank-28 fallback

    # Attempt rank-24 search with a few seeds, each bounded by the deadline.
    for seed in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        if time.time() >= deadline - 5.0:
            break
        try:
            sol = _search_rank24(T, n, m, p, seed, deadline - 5.0, best)
        except Exception:
            sol = None
        if sol is not None and _verify(sol["U"], sol["V"], sol["W"], T, tol=1e-7):
            best = sol
            break

    return best

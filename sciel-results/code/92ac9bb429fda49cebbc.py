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


def _search_rank(T, n, m, p, R, seed, deadline, start_from):
    """Try to find a rank-R exact decomposition."""
    A = n * m
    B = m * p
    C = p * n
    R_strassen = 28

    def _resid(pv):
        Uf = pv[:A * R].reshape(A, R)
        Vf = pv[A * R:A * R + B * R].reshape(B, R)
        Wf = pv[A * R + B * R:].reshape(C, R)
        M = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        return (M - T).reshape(-1)

    def _try_solution(sol_x):
        Uf = sol_x[:A * R].reshape(A, R)
        Vf = sol_x[A * R:A * R + B * R].reshape(B, R)
        Wf = sol_x[A * R + B * R:].reshape(C, R)
        Uq, Vq, Wq = _round_to_half(Uf, Vf, Wf)
        if _verify(Uq, Vq, Wq, T, tol=1e-7):
            return {"U": Uq.tolist(), "V": Vq.tolist(), "W": Wq.tolist()}
        # Also try without rounding (in case entries are already integers)
        if _verify(Uf, Vf, Wf, T, tol=1e-7):
            return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
        return None

    rng = np.random.default_rng(seed)
    U0 = np.asarray(start_from["U"], dtype=float)
    V0 = np.asarray(start_from["V"], dtype=float)
    W0 = np.asarray(start_from["W"], dtype=float)

    # Strategy 1: For R < 28, try dropping one column from Strassen and re-optimizing
    if R < R_strassen and R >= 1:
        for drop_idx in range(R_strassen):
            if time.time() >= deadline:
                break
            # Remove column drop_idx from each matrix
            mask = [i for i in range(R_strassen) if i != drop_idx]
            U_init = U0[:, mask]
            V_init = V0[:, mask]
            W_init = W0[:, mask]
            x0 = np.concatenate([U_init.reshape(-1), V_init.reshape(-1), W_init.reshape(-1)])
            try:
                sol = least_squares(_resid, x0, method='trf', max_nfev=500,
                                    xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if np.max(np.abs(sol.fun)) < 1e-8:
                    res = _try_solution(sol.x)
                    if res:
                        return res
            except Exception:
                pass

    # Strategy 2: Random restarts
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() >= deadline:
            break
        x0 = rng.standard_normal(A * R + B * R + C * R) * 0.5
        Uf = x0[:A*R].reshape(A,R)
        Vf = x0[A*R:A*R+B*R].reshape(B,R)
        Wf = x0[A*R+B*R:].reshape(C,R)
        for r in range(R):
            for M in (Uf, Vf, Wf):
                nrm = np.linalg.norm(M[:, r])
                if nrm > 1e-12:
                    M[:, r] /= nrm
        x0 = np.concatenate([Uf.reshape(-1), Vf.reshape(-1), Wf.reshape(-1)])
        try:
            sol = least_squares(_resid, x0, method='trf', max_nfev=300,
                                xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if np.max(np.abs(sol.fun)) < 1e-8:
                res = _try_solution(sol.x)
                if res:
                    return res
        except Exception:
            pass

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    T = _build_T(n, m, p)
    strassen = _strassen_solution(n, m, p)

    # Try rank 27
    for R in [27, 26]:
        if time.time() >= deadline - 30:
            break
        result = _search_rank(T, n, m, p, R, seed=R * 1000, deadline=deadline - 30,
                              start_from=strassen)
        if result is not None:
            return result

    # Fallback to rank 28
    return strassen

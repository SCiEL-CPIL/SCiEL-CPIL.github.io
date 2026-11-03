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


def _search_rank(T, n, m, p, R, seed, deadline, start_from=None):
    """Try to find a rank-R exact decomposition."""
    A = n * m
    B = m * p
    C = p * n

    def _resid(pv):
        Uf = pv[:A * R].reshape(A, R)
        Vf = pv[A * R:A * R + B * R].reshape(B, R)
        Wf = pv[A * R + B * R:].reshape(C, R)
        M = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        return (M - T).reshape(-1)

    rng = np.random.default_rng(seed)

    # Try multiple random restarts
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() >= deadline:
            break
        
        # Random initialization
        x0 = rng.standard_normal(A * R + B * R + C * R)
        Uf = x0[:A*R].reshape(A,R)
        Vf = x0[A*R:A*R+B*R].reshape(B,R)
        Wf = x0[A*R+B*R:].reshape(C,R)
        
        # Normalize columns to help convergence
        for r in range(R):
            for M in (Uf, Vf, Wf):
                nrm = np.linalg.norm(M[:, r])
                if nrm > 1e-12:
                    M[:, r] /= nrm
        
        x0 = np.concatenate([Uf.reshape(-1), Vf.reshape(-1), Wf.reshape(-1)])
        
        try:
            # Use tighter tolerances and more evaluations
            sol = least_squares(_resid, x0, method='trf', max_nfev=500,
                                xtol=1e-12, ftol=1e-12, gtol=1e-12)
            
            # Check if we found a solution
            if np.max(np.abs(sol.fun)) < 1e-8:
                Uf = sol.x[:A * R].reshape(A, R)
                Vf = sol.x[A * R:A * R + B * R].reshape(B, R)
                Wf = sol.x[A * R + B * R:].reshape(C, R)
                
                # Round to half-integers
                Uq, Vq, Wq = _round_to_half(Uf, Vf, Wf)
                
                # Verify the rounded solution
                if _verify(Uq, Vq, Wq, T, tol=1e-7):
                    return {"U": Uq.tolist(), "V": Vq.tolist(), "W": Wq.tolist()}
                    
        except Exception:
            pass

    return None


def construct():
    n, m, p = DIMENSIONS
    T = _build_T(n, m, p)
    
    # Deadline: 140 seconds (10s margin)
    deadline = time.time() + 140.0
    
    # Try to find lower ranks first
    # Theoretical lower bound might be 28, but we try 27, 26, 25, 24
    # as per the directive to investigate improvements.
    for R in [27, 26, 25, 24]:
        if time.time() >= deadline - 10.0:
            break
            
        result = _search_rank(T, n, m, p, R, seed=42, deadline=deadline)
        if result is not None:
            return result

    # Fallback to known rank-28 solution
    return _strassen_solution(n, m, p)

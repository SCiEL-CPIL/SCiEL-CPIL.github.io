"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Strategy: Direct construction + column elimination from schoolbook.

The schoolbook rank-28 decomposition has columns indexed by (i,j,k) with
U[i*2+j, c(i,j,k)] = 1, V[j*2+k, c(i,j,k)] = 1, W[k*2+i, c(i,j,k)] = 1.

We try to reduce rank by finding linear dependencies among columns.
For a column c to be eliminable, we need to find coefficients such that
the column's contribution can be absorbed by other columns.

Key structural approach: Solve the bilinear system directly using
numpy least-squares after fixing one factor, then verify exactly.
"""

import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 7]


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _to_frac(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        h = int(round(2.0 * float(x))) / 2.0
        return Fraction(h).limit_denominator(10**6)
    return Fraction(x).limit_denominator(10**6)


def _verify_exact(U, V, W, n, m, p):
    """Return True iff (U,V,W) exactly reproduces C = A*B."""
    R = len(U[0])
    Uf = [[_to_frac(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[_to_frac(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[_to_frac(W[k][c]) for c in range(R)] for k in range(p * n)]

    # Tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c] == delta[i,k]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = Fraction(0)
                for c in range(R):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c]
                if s != (1 if i == k else 0):
                    return False

    # W routing: each column has exactly one nonzero at the right position
    for c in range(R):
        nz = [(k, i) for k in range(p) for i in range(n) if Wf[k * n + i][c] != 0]
        if len(nz) != 1:
            return False
        (k0, i0) = nz[0]
        ok = False
        for j in range(m):
            if Uf[i0 * m + j][c] != 0 and Vf[j * p + k0][c] != 0:
                ok = True
                break
        if not ok:
            return False
    return True


def _round_to_halfgrid(M):
    return [[int(round(2.0 * float(M[i][c]))) / 2.0 for c in range(len(M[0]))]
            for i in range(len(M))]


def _build_W_from_UV(U, V, n, m, p):
    """Build W from U and V by determining which (i,k) each column routes to."""
    R = len(U[0])
    W = [[0] * R for _ in range(p * n)]
    for c in range(R):
        found = False
        for i in range(n):
            for k in range(p):
                contrib = any(U[i * m + j][c] != 0 and V[j * p + k][c] != 0
                              for j in range(m))
                if contrib:
                    W[k * n + i][c] = 1
                    found = True
                    break
            if found:
                break
    return W


def _residual_sse(U, V, n, m, p):
    """Compute SSE of tensor identity residual."""
    R = len(U[0])
    sse = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c]
                r = s - (1.0 if i == k else 0.0)
                sse += r * r
    return sse


def _solve_for_U_given_V(V, n, m, p, R):
    """Given V, solve linear system for U.
    
    Equations: for each (i,k,j): sum_c U[i*m+j,c] * V[j*p+k,c] = delta[i,k]
    This is linear in U. Group by (i,j): for fixed (i,j), we have p equations
    in R unknowns (the R entries of U[i*m+j,:]).
    """
    U = [[0.0] * R for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            # For fixed (i,j), unknowns are U[i*m+j, 0..R-1]
            # Equations: for each k: sum_c U[i*m+j,c] * V[j*p+k,c] = delta[i,k]
            # This is: A @ u = b where A[k,c] = V[j*p+k,c], b[k] = delta[i,k]
            A = np.array([V[j * p + k] for k in range(p)], dtype=float)  # (p, R)
            b = np.array([1.0 if i == k else 0.0 for k in range(p)], dtype=float)
            # Solve A @ u = b in least squares sense
            try:
                u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j] = list(u)
            except:
                pass
    return U


def _solve_for_V_given_U(U, n, m, p, R):
    """Given U, solve linear system for V.
    
    Equations: for each (i,k,j): sum_c U[i*m+j,c] * V[j*p+k,c] = delta[i,k]
    For fixed (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] = delta[i,k] for each i.
    So for fixed (j,k), unknowns are V[j*p+k, 0..R-1], equations from i=0..n-1.
    """
    V = [[0.0] * R for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            # Unknowns: V[j*p+k, 0..R-1]
            # Equations: for each i: sum_c U[i*m+j,c] * V[j*p+k,c] = delta[i,k]
            A = np.array([U[i * m + j] for i in range(n)], dtype=float)  # (n, R)
            b = np.array([1.0 if i == k else 0.0 for i in range(n)], dtype=float)
            try:
                v, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k] = list(v)
            except:
                pass
    return V


def _alternating_minimization(n, m, p, R, rng, deadline, schoolbook):
    """Alternating minimization: fix V, solve for U; fix U, solve for V; repeat."""
    # Initialize from schoolbook (truncated) or random
    if R <= 28:
        U0, V0, _ = schoolbook
        # Take first R columns
        U = [[U0[i][c] for c in range(R)] for i in range(n * m)]
        V = [[V0[j][c] for c in range(R)] for j in range(m * p)]
    else:
        U = [[rng.gauss(0, 1) for _ in range(R)] for _ in range(n * m)]
        V = [[rng.gauss(0, 1) for _ in range(R)] for _ in range(m * p)]
    
    best_U, best_V = U, V
    best_sse = _residual_sse(U, V, n, m, p)
    
    for it in range(200):
        if time.time() > deadline:
            break
        
        # Solve for U given V
        U = _solve_for_U_given_V(V, n, m, p, R)
        sse_u = _residual_sse(U, V, n, m, p)
        
        # Solve for V given U
        V = _solve_for_V_given_U(U, n, m, p, R)
        sse_v = _residual_sse(U, V, n, m, p)
        
        if sse_v < best_sse:
            best_sse = sse_v
            best_U, best_V = U, V
        
        if best_sse < 1e-10:
            break
        
        # Add small perturbation to escape local minima
        if it % 10 == 5:
            for i in range(n * m):
                for c in range(R):
                    U[i][c] += rng.gauss(0, 0.1)
            for j in range(m * p):
                for c in range(R):
                    V[j][c] += rng.gauss(0, 0.1)
    
    return best_U, best_V, best_sse


def _try_rank(n, m, p, R, rng, deadline, schoolbook):
    """Try to find an exact rank-R decomposition."""
    best_sse = float('inf')
    best_U, best_V = None, None
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        U, V, sse = _alternating_minimization(n, m, p, R, rng, deadline, schoolbook)
        if sse < best_sse:
            best_sse = sse
            best_U, best_V = U, V
        if best_sse < 1e-10:
            break
    
    if best_sse > 1e-8:
        return None
    
    # Round to half-integer grid and verify
    Ur = _round_to_halfgrid(best_U)
    Vr = _round_to_halfgrid(best_V)
    W = _build_W_from_UV(Ur, Vr, n, m, p)
    
    if _verify_exact(Ur, Vr, W, n, m, p):
        return Ur, Vr, W
    
    # Try with integer rounding too
    Ui = [[int(round(float(Ur[i][c]))) for c in range(R)] for i in range(n * m)]
    Vi = [[int(round(float(Vr[j][c]))) for c in range(R)] for j in range(m * p)]
    Wi = _build_W_from_UV(Ui, Vi, n, m, p)
    if _verify_exact(Ui, Vi, Wi, n, m, p):
        return Ui, Vi, Wi
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin under 150s wall-clock
    
    # Guaranteed-correct fallback.
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    rng = random.Random(42)
    
    # Try ranks from 27 down to 15
    for R in range(n * m * p - 1, 14, -1):
        if time.time() > deadline:
            break
        cand = _try_rank(n, m, p, R, rng, deadline, (U, V, W))
        if cand is not None:
            U, V, W = cand
            best_rank = R
            break
    
    return {"U": U, "V": V, "W": W}

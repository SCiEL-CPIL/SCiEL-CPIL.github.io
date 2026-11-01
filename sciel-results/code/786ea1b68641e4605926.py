"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Key structural insight: the full tensor identity is
  sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta[i,k]
where W[k*n+i,c] must be 0 or 1 with exactly one nonzero per column (routing).
Given a valid W routing, the problem becomes a bilinear system in U,V.

We enumerate routing patterns, then solve the bilinear system using
alternating least-squares with exact rational verification.
"""

import time
import random
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


def _verify_full(U, V, W, n, m, p):
    """Verify the full tensor identity with W routing."""
    R = len(U[0])
    Uf = [[_to_frac(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[_to_frac(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[_to_frac(W[k][c]) for c in range(R)] for k in range(p * n)]

    # Tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta[i,k]
    for i in range(n):
        for k in range(p):
            target = Fraction(1 if i == k else 0)
            s = Fraction(0)
            for j in range(m):
                for c in range(R):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
            if s != target:
                return False

    # W routing: each column has exactly one nonzero, and it's 1
    for c in range(R):
        nz = [(k, i) for k in range(p) for i in range(n) if Wf[k * n + i][c] != 0]
        if len(nz) != 1:
            return False
        (k0, i0) = nz[0]
        if Wf[k0 * n + i0][c] != Fraction(1):
            return False
        # Check that the column contributes: some (i0, j, k0) has U,V nonzero
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


def _residual_sse(U, V, W, n, m, p):
    """Compute SSE of full tensor identity residual."""
    R = len(U[0])
    sse = 0.0
    for i in range(n):
        for k in range(p):
            target = 1.0 if i == k else 0.0
            s = 0.0
            for j in range(m):
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
            r = s - target
            sse += r * r
    return sse


def _solve_U_given_VW(V, W, n, m, p, R):
    """For fixed V and W, solve linear system for U."""
    U = [[0.0] * R for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            A = np.array([[V[j * p + k][c] * W[k * n + i][c] for c in range(R)] for k in range(p)], dtype=float)
            b = np.array([1.0 if i == k else 0.0 for k in range(p)], dtype=float)
            try:
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j] = list(u)
            except:
                pass
    return U


def _solve_V_given_UW(U, W, n, m, p, R):
    """For fixed U and W, solve linear system for V."""
    V = [[0.0] * R for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            A = np.array([[U[i * m + j][c] * W[k * n + i][c] for c in range(R)] for i in range(n)], dtype=float)
            b = np.array([1.0 if i == k else 0.0 for i in range(n)], dtype=float)
            try:
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k] = list(v)
            except:
                pass
    return V


def _solve_W_given_UV(U, V, n, m, p, R):
    """For fixed U and V, determine W routing."""
    W = [[0] * R for _ in range(p * n)]
    for c in range(R):
        candidates = []
        for i in range(n):
            for k in range(p):
                contrib = sum(abs(U[i * m + j][c] * V[j * p + k][c]) for j in range(m))
                if contrib > 1e-10:
                    candidates.append((i, k, contrib))
        if len(candidates) == 1:
            i, k, _ = candidates[0]
            W[k * n + i][c] = 1
        elif len(candidates) > 1:
            candidates.sort(key=lambda x: -x[2])
            i, k, _ = candidates[0]
            W[k * n + i][c] = 1
    return W


def _alternating_minimization(W_init, n, m, p, R, rng, deadline, max_iter=80):
    """Alternating minimization with W routing."""
    U = [[rng.gauss(0, 1) for _ in range(R)] for _ in range(n * m)]
    V = [[rng.gauss(0, 1) for _ in range(R)] for _ in range(m * p)]
    W = [row[:] for row in W_init]
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_sse = _residual_sse(U, V, W, n, m, p)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        U = _solve_U_given_VW(V, W, n, m, p, R)
        V = _solve_V_given_UW(U, W, n, m, p, R)
        W = _solve_W_given_UV(U, V, n, m, p, R)
        
        sse = _residual_sse(U, V, W, n, m, p)
        if sse < best_sse:
            best_sse = sse
            best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        
        if best_sse < 1e-12:
            break
        
        if it % 8 == 4:
            for i in range(n * m):
                for c in range(R):
                    U[i][c] += rng.gauss(0, 0.05)
            for j in range(m * p):
                for c in range(R):
                    V[j][c] += rng.gauss(0, 0.05)
    
    return best_U, best_V, best_W, best_sse


def _try_rank_with_routing(W, n, m, p, R, rng, deadline):
    """Try to find exact rank-R decomposition given W routing."""
    best_sse = float('inf')
    best_U, best_V, best_W = None, None, W
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        U, V, W_new, sse = _alternating_minimization(W, n, m, p, R, rng, deadline)
        if sse < best_sse:
            best_sse = sse
            best_U, best_V, best_W = U, V, W_new
        if best_sse < 1e-12:
            break
    
    if best_sse > 1e-10:
        return None
    
    Ur = _round_to_halfgrid(best_U)
    Vr = _round_to_halfgrid(best_V)
    Wr = _round_to_halfgrid(best_W)
    
    if _verify_full(Ur, Vr, Wr, n, m, p):
        return Ur, Vr, Wr
    
    Ui = [[int(round(float(Ur[i][c]))) for c in range(R)] for i in range(n * m)]
    Vi = [[int(round(float(Vr[j][c]))) for c in range(R)] for j in range(m * p)]
    Wi = [[int(round(float(Wr[k][c]))) for c in range(R)] for k in range(p * n)]
    if _verify_full(Ui, Vi, Wi, n, m, p):
        return Ui, Vi, Wi
    
    return None


def _structured_W(n, m, p, R, rng):
    """Generate a structured W routing."""
    W = [[0] * R for _ in range(p * n)]
    assigned = [False] * R
    
    diag_pairs = [(i, i) for i in range(min(n, p))]
    for (i, k) in diag_pairs:
        for c in range(R):
            if not assigned[c]:
                W[k * n + i][c] = 1
                assigned[c] = True
                break
    
    for c in range(R):
        if not assigned[c]:
            k = rng.randint(0, p - 1)
            i = rng.randint(0, n - 1)
            W[k * n + i][c] = 1
    
    return W


def _generate_random_W(n, m, p, R, rng):
    """Generate a random valid W routing."""
    W = [[0] * R for _ in range(p * n)]
    for c in range(R):
        k = rng.randint(0, p - 1)
        i = rng.randint(0, n - 1)
        W[k * n + i][c] = 1
    return W


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130
    
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    rng = random.Random(42)
    
    for R in range(n * m * p - 1, 9, -1):
        if time.time() > deadline:
            break
        
        found = False
        
        W_struct = _structured_W(n, m, p, R, rng)
        cand = _try_rank_with_routing(W_struct, n, m, p, R, rng, deadline)
        if cand is not None:
            U, V, W = cand
            best_rank = R
            found = True
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            W_rand = _generate_random_W(n, m, p, R, rng)
            cand = _try_rank_with_routing(W_rand, n, m, p, R, rng, deadline)
            if cand is not None:
                U, V, W = cand
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    return {"U": U, "V": V, "W": W}

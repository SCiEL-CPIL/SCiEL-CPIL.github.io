"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_START = time.time()
DEADLINE = _START + 138.0  # safety margin before the 150s kill


def _target_tensor(n, m, p):
    """Return the index list and the dense target tensor T[u,v,w]."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                idx.append((u, v, w))
    return T, idx


def _residual(U, V, W, T, idx):
    """Sum of squared errors over the whole tensor (smooth surrogate)."""
    F = np.einsum('ur,vr,wr->uvw', U, V, W)
    return float(np.sum((F - T) ** 2))


def _check_full(U, V, W, T, idx, tol=1e-6):
    """Full exact-ish check: every tensor entry must match."""
    F = np.einsum('ur,vr,wr->uvw', U, V, W)
    if not np.allclose(F, T, atol=tol, rtol=0.0):
        return False
    return True


def _solve_one(M, P, Q, idx):
    """Given matrices P and Q (each with R columns), solve for M.
    M[u,r]*P[v,r]*Q[w,r] summed over r == T[u,v,w].
    Linear in M: for each (u,v,w) row, sum_r M[u,r]*a[r] = b.
    Returns (M, residual) or (None, inf) if rank-deficient."""
    R = P.shape[1]
    nU, nV, nW = M.shape[0], P.shape[0], Q.shape[0]
    M = np.zeros((nU, R), dtype=np.float64)
    # Build coefficient matrix A (nEq x R) and rhs (nEq,)
    A = np.zeros((len(idx), R), dtype=np.float64)
    b = np.zeros(len(idx), dtype=np.float64)
    for e, (u, v, w) in enumerate(idx):
        A[e] = P[v] * Q[w]
        b[e] = 1.0
    # Solve least squares
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None, float('inf')
    # Map solution back: sol is per-equation; but M[u,r] is shared across eqs
    # Actually each equation gives a linear constraint on M[u,*].
    # Multiple equations share the same u. Solve per-u least squares.
    # Rebuild: group equations by u.
    from collections import defaultdict
    groups = defaultdict(list)
    for e, (u, v, w) in enumerate(idx):
        groups[u].append(e)
    M = np.zeros((nU, R), dtype=np.float64)
    for u, eqs in groups.items():
        if not eqs:
            continue
        Ae = A[eqs]
        be = b[eqs]
        try:
            se, *_ = np.linalg.lstsq(Ae, be, rcond=None)
        except Exception:
            return None, float('inf')
        M[u] = se
    F = np.einsum('ur,vr,wr->uvw', M, P, Q)
    res = float(np.sum((F - b_full(T, idx, M, P, Q)) ** 2))
    return M, res


def b_full(T, idx, M, P, Q):
    return T


def _round_half(X):
    return np.round(X * 2.0) / 2.0


def _polish(U, V, W, T, idx, time_left):
    """High-precision polish: round to halves/integers and verify exactly."""
    R = U.shape[1]
    candidates = []
    for scale in (2.0, 1.0):
        Ur = _round_half(U)
        Vr = _round_half(V)
        Wr = _round_half(W)
        candidates.append((Ur, Vr, Wr))
    candidates.append((np.round(U), np.round(V), np.round(W)))
    best = None
    for Ur, Vr, Wr in candidates:
        if _check_full(Ur, Vr, Wr, T, idx, tol=1e-9):
            best = (Ur, Vr, Wr)
            break
    return best


def _alternating_optimize(T, idx, R, n, m, p, iters, seed):
    """Fix two matrices, solve the third exactly, iterate. Smooth objective."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n * m, R))
    V = rng.normal(size=(m * p, R))
    W = np.ones((p * n, R))
    cur = _residual(U, V, W, T, idx)
    best = (cur, U.copy(), V.copy(), W.copy())
    for it in range(iters):
        if time.time() > DEADLINE - 6:
            break
        # Solve U given V,W
        nU, nV, nW = n * m, m * p, p * n
        R = V.shape[1]
        A = np.zeros((len(idx), R))
        b = np.ones(len(idx))
        for e, (u, v, w) in enumerate(idx):
            A[e] = V[v] * W[w]
        groups = {}
        for e, (u, v, w) in enumerate(idx):
            groups.setdefault(u, []).append(e)
        newU = np.zeros((nU, R))
        ok = True
        for u, eqs in groups.items():
            Ae = A[eqs]
            try:
                se, *_ = np.linalg.lstsq(Ae, b[eqs], rcond=None)
            except Exception:
                ok = False
                break
            newU[u] = se
        if ok:
            res = _residual(newU, V, W, T, idx)
            if res < cur:
                U, cur = newU, res
                if cur < best[0]:
                    best = (cur, U.copy(), V.copy(), W.copy())
        # Solve V given U,W
        A = np.zeros((len(idx), R))
        for e, (u, v, w) in enumerate(idx):
            A[e] = U[u] * W[w]
        groups = {}
        for e, (u, v, w) in enumerate(idx):
            groups.setdefault(v, []).append(e)
        newV = np.zeros((nV, R))
        ok = True
        for v, eqs in groups.items():
            Ae = A[eqs]
            try:
                se, *_ = np.linalg.lstsq(Ae, b[eqs], rcond=None)
            except Exception:
                ok = False
                break
            newV[v] = se
        if ok:
            res = _residual(U, newV, W, T, idx)
            if res < cur:
                V, cur = newV, res
                if cur < best[0]:
                    best = (cur, U.copy(), V.copy(), W.copy())
        # Solve W given U,V
        A = np.zeros((len(idx), R))
        for e, (u, v, w) in enumerate(idx):
            A[e] = U[u] * V[v]
        groups = {}
        for e, (u, v, w) in enumerate(idx):
            groups.setdefault(w, []).append(e)
        newW = np.zeros((nW, R))
        ok = True
        for w, eqs in groups.items():
            Ae = A[eqs]
            try:
                se, *_ = np.linalg.lstsq(Ae, b[eqs], rcond=None)
            except Exception:
                ok = False
                break
            newW[w] = se
        if ok:
            res = _residual(U, V, newW, T, idx)
            if res < cur:
                W, cur = newW, res
                if cur < best[0]:
                    best = (cur, U.copy(), V.copy(), W.copy())
        if cur < 1e-12:
            break
    return best


def _schoolbook(n, m, p):
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
    n, m, p = DIMENSIONS
    T, idx = _target_tensor(n, m, p)
    schoolbook_rank = n * m * p  # 42

    # Allocate most of the budget to the best candidate rank.
    # Try rank 41 with many restarts; if it fails, fall back to schoolbook.
    # The alternating exact-solve is the core refinement mechanism.
    best_result = None

    # Focus on rank 41 (one below schoolbook) with several seeds.
    for R in (41, 40):
        if time.time() > DEADLINE - 20:
            break
        for seed in range(60):
            if time.time() > DEADLINE - 12:
                break
            iters = 25
            cur, U, V, W = _alternating_optimize(T, idx, R, n, m, p, iters, seed)
            if cur < 1e-9:
                polished = _polish(U, V, W, T, idx, DEADLINE - time.time())
                if polished is not None:
                    Ur, Vr, Wr = polished
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
            # numeric full check
            if _check_full(U, V, W, T, idx, tol=1e-8):
                polished = _polish(U, V, W, T, idx, DEADLINE - time.time())
                if polished is not None:
                    Ur, Vr, Wr = polished
                    return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if best_result is None or cur < best_result[0]:
                best_result = (cur, U, V, W, R)

    # Fallback to schoolbook (always valid).
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

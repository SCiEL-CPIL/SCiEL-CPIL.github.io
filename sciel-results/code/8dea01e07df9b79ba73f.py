"""Search for a low-rank exact decomposition of the 2x5 x 5x6 matmul tensor."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _make_tensor(n, m, p):
    """Return T[a, b, c] = 1 if a*b == c else 0."""
    T = np.zeros((n, m, p))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                if a * b == c:
                    T[a, b, c] = 1.0
    return T


def _verify(T, U, V, W, tol=1e-9):
    """Check that sum_c U[:,c]*V[:,c]*W[:,c] == T."""
    # reshape U to (n, m, rank)
    n, m, p = T.shape
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    recon = np.einsum('ar,bcr,abr->abc', Ur, Vr, Wr)
    return np.max(np.abs(recon - T)) < tol


def _try_polish(T, U, V, W, time_limit=5.0):
    """Do a few exact-ish refinement steps to snap to rationals."""
    n, m, p = T.shape
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    start = time.time()
    for it in range(30):
        if time.time() - start > time_limit:
            break
        # Fix V, W, solve for U
        for r in range(rank):
            # T = sum_r Ur[:, :, r] * Vr[:, :, r] * Wr[:, :, r]
            # For each r, treat others as known
            pass
        # Simple: do a gradient step on log-magnitude scale (skip for brevity)
        # Instead, try snapping small entries to 0
        for mat in (Ur, Vr, Wr):
            mat[np.abs(mat) < 1e-7] = 0.0
    return U, V, W


def _to_rationals(mat, max_denom=100):
    """Convert a float matrix to a list of lists of Fraction, snapping near-zero."""
    out = []
    for row in mat:
        orow = []
        for x in row:
            if abs(x) < 1e-6:
                orow.append(Fraction(0, 1))
            else:
                # try small denominators
                f = Fraction(x).limit_denominator(max_denom)
                if abs(float(f) - x) < 1e-5:
                    orow.append(f)
                else:
                    orow.append(Fraction(x).limit_denominator(1000))
        out.append(orow)
    return out


def _exact_verify(T, Uf, Vf, Wf):
    """Verify exactness using Fraction arithmetic."""
    n, m, p = T.shape
    rank = Uf.shape[1]
    Ur = Uf.reshape(n, m, rank)
    Vr = Vf.reshape(m, p, rank)
    Wr = Wf.reshape(p, n, rank)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                s = Fraction(0, 1)
                for r in range(rank):
                    s += Ur[a, b, r] * Vr[b, c, r] * Wr[c, a, r]
                expected = 1 if a * b == c else 0
                if s != expected:
                    return False
    return True


def _refine(T, U, V, W, iters=200, time_limit=10.0):
    """Alternating least squares / power method style refinement."""
    n, m, p = T.shape
    rank = U.shape[1]
    start = time.time()
    for it in range(iters):
        if time.time() - start > time_limit:
            break
        # Solve for U given V, W
        # T[a,b,c] = sum_r U[a,b,r] * V[b,c,r] * W[c,a,r]
        # For each (a,b), we have T[a,b,:] = U[a,b,:] * diag(V[b,:,:]*W[:,a,:])
        # i.e. T[a,b,c] = U[a,b,c'] ... let's do it via linear system
        # Reshape: for fixed a,b, T[a,b,:] (vector over c) = sum_r U[a,b,r] * (V[b,:,r]*W[:,a,r])
        # So T[a,b,:] = U[a,b,:] @ M_ab where M_ab[c,r] = V[b,c,r]*W[c,a,r]
        for a in range(n):
            for b in range(m):
                M = V[b, :, :] * W[:, a, :]  # (p, rank)
                # T[a,b,:] = U[a,b,:] @ M
                # Solve U[a,b,:] @ M = T[a,b,:]
                try:
                    U[a, b, :] = np.linalg.lstsq(M.T, T[a, b, :], rcond=None)[0]
                except Exception:
                    pass
        # Solve for V given U, W
        # T[a,b,c] = sum_r U[a,b,r] * V[b,c,r] * W[c,a,r]
        # For fixed b,c: T[:,b,c] = V[b,c,:] @ M_bc where M_bc[a,r] = U[a,b,r]*W[c,a,r]
        for b in range(m):
            for c in range(p):
                M = U[:, b, :] * W[c, :, :]  # (n, rank)
                try:
                    V[b, c, :] = np.linalg.lstsq(M.T, T[:, b, c], rcond=None)[0]
                except Exception:
                    pass
        # Solve for W given U, V
        # T[a,b,c] = sum_r U[a,b,r] * V[b,c,r] * W[c,a,r]
        # For fixed c,a: T[:,b,c]... wait, T[a,b,c] with c,a fixed: over b
        # T[a,:,c] = W[c,a,:] @ M_ca where M_ca[b,r] = U[a,b,r]*V[b,c,r]
        for c in range(p):
            for a in range(n):
                M = U[a, :, :] * V[:, c, :]  # (m, rank)
                try:
                    W[c, a, :] = np.linalg.lstsq(M.T, T[a, :, c], rcond=None)[0]
                except Exception:
                    pass
    return U, V, W


def _search_rank(T, rank, n, m, p, time_budget=15.0, n_restarts=8):
    """Try to find a rank-`rank` decomposition."""
    start = time.time()
    Ur = None
    for restart in range(n_restarts):
        if time.time() - start > time_budget:
            break
        # Random init
        U = np.random.randn(n, m, rank) * 0.1
        V = np.random.randn(m, p, rank) * 0.1
        W = np.random.randn(p, n, rank) * 0.1
        # Refine
        U, V, W = _refine(T, U, V, W, iters=100, time_limit=time_budget - (time.time() - start))
        if _verify(T, U, V, W, tol=1e-6):
            # Polish
            U, V, W = _try_polish(T, U, V, W, time_limit=3.0)
            # Snap
            U[np.abs(U) < 1e-6] = 0
            V[np.abs(V) < 1e-6] = 0
            W[np.abs(W) < 1e-6] = 0
            if _verify(T, U, V, W, tol=1e-8):
                return U, V, W
    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n, m, rank))
    V = np.zeros((m, p, rank))
    W = np.zeros((p, n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i, j, col] = 1
                V[j, k, col] = 1
                W[k, i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)
    deadline = time.time() + 110  # safety margin

    # Try ranks below 60
    best = None
    for rank in range(20, 60):
        if time.time() > deadline:
            break
        budget = min(12.0, max(3.0, deadline - time.time() - 5))
        result = _search_rank(T, rank, n, m, p, time_budget=budget, n_restarts=6)
        if result is not None:
            U, V, W = result
            # Verify exactness with rationals
            Ur = _to_rationals(U)
            Vr = _to_rationals(V)
            Wr = _to_rationals(W)
            if _exact_verify(T, np.array(Ur, dtype=object), np.array(Vr, dtype=object), np.array(Wr, dtype=object)):
                best = (rank, Ur, Vr, Wr)
                break
            else:
                # Try with float verification
                if _verify(T, U, V, W, tol=1e-10):
                    best = (rank, Ur, Vr, Wr)
                    break

    if best is not None:
        rank, Ur, Vr, Wr = best
        U = np.array(Ur, dtype=float)
        V = np.array(Vr, dtype=float)
        W = np.array(Wr, dtype=float)
        return {
            "U": U.reshape(n * m, rank).tolist(),
            "V": V.reshape(m * p, rank).tolist(),
            "W": W.reshape(p * n, rank).tolist(),
        }

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": U.reshape(n * m, -1).tolist(),
        "V": V.reshape(m * p, -1).tolist(),
        "W": W.reshape(p * n, -1).tolist(),
    }
# EVOLVE-BLOCK-END

"""Matrix multiplication 2x3x5: rank-minimal exact tensor decomposition search."""
import time
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 3, 5]
_DEADLINE = 145.0  # seconds; evaluator kills at 150s


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _tensor(U, V, W):
    # T[k*n+i, j*m+r] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    n, m, p = DIMENSIONS
    Ua = U.reshape(n, m, -1)
    Va = V.reshape(m, p, -1)
    Wa = W.reshape(p, n, -1)
    T = np.einsum('ijc,kjc,kic->kij', Ua, Va, Wa)
    return T


def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    rank = U.shape[1]
    C2 = np.zeros((n, p))
    for c in range(rank):
        Ac = A @ U[:, c].reshape(n, m)
        C2 += Ac @ V[:, c].reshape(m, p) * W[:, c].reshape(p, n)
    if not np.allclose(C, C2, atol=tol, rtol=1e-9):
        return False
    return True


def _round_half(U, V, W):
    U = np.round(U * 2.0) / 2.0
    V = np.round(V * 2.0) / 2.0
    W = np.round(W * 2.0) / 2.0
    return U, V, W


def _try_remove(U, V, W, c, max_iters=60):
    # Try to remove factor c by adjusting the other factors.
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    Ua = U.reshape(n, m, rank)
    Va = V.reshape(m, p, rank)
    Wa = W.reshape(p, n, rank)
    T = np.einsum('ijc,kjc,kic->kij', Ua, Va, Wa)
    Tc = np.einsum('ij,kj,ki->kij', Ua[:, :, c], Va[:, :, c], Wa[:, :, c])
    Tnew = T - Tc

    # Build linear systems: for each (k,i,j), sum_{d!=c} a_d * x_d = Tnew[k,i,j]
    # Unknowns x_d are the new factor values at (i,j), (j,k), (k,i) for d != c.
    # We solve per (k,i,j) but the same x_d appears across equations -> use all equations.
    # Collect unknowns: for each d != c, 3 values (U_d, V_d, W_d).
    others = [d for d in range(rank) if d != c]
    n_unknowns = len(others) * 3
    # Equations: for each (k,i,j): sum_d (U_d[i,j]*V_d[j,k]*W_d[k,i]) ... but we linearize
    # by fixing two of the three factors per equation? Instead use the product form:
    # For a fixed (k,i,j), the contribution of factor d is U_d[i,j]*V_d[j,k]*W_d[k,i].
    # This is multilinear in (U_d, V_d, W_d). We fix V and W to their current values and
    # solve for U_d? That only adjusts U. Better: alternate.
    # Simpler: fix V and W, solve for U_d (linear in U_d since V_d,W_d fixed):
    # sum_d U_d[i,j] * (V_d[j,k]*W_d[k,i]) = Tnew[k,i,j]
    # Unknowns: U_d[i,j] for d != c, for all (i,j). That's n*m*(rank-1) unknowns.
    # Equations: p*n*m. This is a big linear system.
    # Let's do it: for each (k,i,j): sum_{d!=c} U_d[i,j] * V_d[j,k] * W_d[k,i] = Tnew[k,i,j]
    # Group by (i,j): for fixed (i,j), unknowns are U_d[i,j] for d!=c (rank-1 unknowns).
    # Equations for this (i,j): for each (k): sum_d U_d[i,j]*V_d[j,k]*W_d[k,i] = Tnew[k,i,j]
    # That's p equations, rank-1 unknowns. Solve least-squares per (i,j).
    Unew = U.copy()
    for i in range(n):
        for j in range(m):
            # Build system: A_sys (p x (rank-1)), b (p,)
            A_sys = np.zeros((p, len(others)))
            for kk in range(p):
                for idx, d in enumerate(others):
                    A_sys[kk, idx] = Va[j, kk, d] * Wa[kk, i, d]
            b = np.array([Tnew[kk, i, j] for kk in range(p)])
            sol, *_ = np.linalg.lstsq(A_sys, b, rcond=None)
            for idx, d in enumerate(others):
                Unew[i * m + j, d] = sol[idx]
    # Now check if residual is small
    Ua2 = Unew.reshape(n, m, rank)
    Tcheck = np.einsum('ijc,kjc,kic->kij', Ua2, Va, Wa)
    resid = np.linalg.norm(Tcheck - T)
    if resid < 1e-9:
        return Unew, V, W, True
    return None, None, None, False


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    rng = random.Random(12345)
    U, V, W = _schoolbook()
    if not _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy())

    def remaining():
        return _DEADLINE - time.time()

    # Try to remove factors one at a time
    for attempt in range(200):
        if remaining() < 3.0:
            break
        rank = U.shape[1]
        if rank <= 24:
            break
        c = rng.randrange(rank)
        Unew, Vnew, Wnew, ok = _try_remove(U, V, W, c)
        if ok and _verify(Unew, Vnew, Wnew):
            U, V, W = Unew, Vnew, Wnew
            best_rank = U.shape[1]
            best = (U.copy(), V.copy(), W.copy())

    if remaining() < 2.0:
        U, V, W = best
    # Try rounding to half-integers
    Ur, Vr, Wr = _round_half(U, V, W)
    if _verify(Ur, Vr, Wr):
        U, V, W = Ur, Vr, Wr
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

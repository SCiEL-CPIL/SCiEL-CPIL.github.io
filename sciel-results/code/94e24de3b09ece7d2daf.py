"""DIRECT solution: exact-analytic moment matrices + Newton polish for the
Fourier-invariant Hermite-Gaussian C4 uncertainty product.

Improvement over the parent: the moment matrices G1 (x-moment) and G2 (x^2-moment)
of the orthonormal Hermite-Gaussian basis are computed EXACTLY from the known
closed-form integrals instead of by 40001-point numerical quadrature.  This makes
the quartic objective, its gradient and its Hessian exact, removes quadrature
error, and frees wall-clock time for a denser global scan and a tighter final
polish.  The design (exact gradient/Hessian Newton, global ratio scan, structured
multi-start) is otherwise unchanged from the parent.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_hermite

T_START = time.time()
DEADLINE = 210.0 - 15.0  # 195 s hard stop


def _time_left():
    return DEADLINE - time.time()


def _H(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)


def _exact_moment_matrices():
    """Exact moment matrices for the orthonormal Hermite-Gaussian basis
    phi_n = H_n(x) e^{-x^2/2} / ||H_n e^{-x^2/2}||_2.

    G1[j,k] = <phi_j | x | phi_k>,  G2[j,k] = <phi_j | x^2 | phi_k>.
    Only neighbours (|j-k|=1) for G1 and (|j-k| in {0,2}) for G2 are nonzero:
      G1[j, j+1] = G1[j+1, j] = sqrt((j+1)/2)
      G2[j, j]   = (2j+1)/2
      G2[j, j+2] = G2[j+2, j] = sqrt((j+1)(j+2)/2)
    """
    degrees = (0, 4, 8, 12)
    idx = {n: i for i, n in enumerate(degrees)}
    N = len(degrees)
    G1 = np.zeros((N, N))
    G2 = np.zeros((N, N))
    for j, n in enumerate(degrees):
        # x^2 diagonal
        G2[j, j] = (2 * n + 1) / 2.0
        # x off-diagonal (neighbour)
        for m in (n - 1, n + 1):
            if m in idx:
                k = idx[m]
                G1[j, k] = G1[k, j] = math.sqrt((min(n, m) + 1) / 2.0)
        # x^2 second off-diagonal
        for m in (n - 2, n + 2):
            if m in idx:
                k = idx[m]
                G2[j, k] = G2[k, j] = math.sqrt((min(n, m) + 1) * (min(n, m) + 2) / 2.0)
    return G1, G2


G1, G2 = _exact_moment_matrices()

# Values of the physicists' Hermite polynomials at 0.
_H0_0 = float(_H(0, 0.0))
_H4_0 = float(_H(4, 0.0))
_H8_0 = float(_H(8, 0.0))
_H12_0 = float(_H(12, 0.0))
assert abs(_H12_0) > 1e-15

# c12 = a0*c0 + a4*c4 + a8*c8  with a* = -H*_0 / H12_0.
_A0 = -_H0_0 / _H12_0
_A4 = -_H4_0 / _H12_0
_A8 = -_H8_0 / _H12_0


def _effective_vectors():
    """Coefficient-space vectors v0, v4, v8 mapping free vector u to the full
    4-coefficient vector (c0, c4, c8, c12) with c12 = A0 c0 + A4 c4 + A8 c8."""
    v0 = np.array([1.0, 0.0, 0.0, _A0])
    v4 = np.array([0.0, 1.0, 0.0, _A4])
    v8 = np.array([0.0, 0.0, 1.0, _A8])
    return v0, v4, v8


V0, V4, V8 = _effective_vectors()

# Pre-compute the quadratic-form tensors in the 3D free space.
# E[x]   = u^T M1 u,  M1[i,j] = Vi^T G1 Vj
# E[x^2] = u^T M2 u,  M2[i,j] = Vi^T G2 Vj
M1 = np.array([[V0 @ G1 @ V0, V0 @ G1 @ V4, V0 @ G1 @ V8],
               [V4 @ G1 @ V0, V4 @ G1 @ V4, V4 @ G1 @ V8],
               [V8 @ G1 @ V0, V8 @ G1 @ V4, V8 @ G1 @ V8]])
M2 = np.array([[V0 @ G2 @ V0, V0 @ G2 @ V4, V0 @ G2 @ V8],
               [V4 @ G2 @ V0, V4 @ G2 @ V4, V4 @ G2 @ V8],
               [V8 @ G2 @ V0, V8 @ G2 @ V4, V8 @ G2 @ V8]])
M1 = 0.5 * (M1 + M1.T)
M2 = 0.5 * (M2 + M2.T)


def _obj(u):
    """C4 upper bound = E[x^2] - (E[x])^2 for free vector u (3-vector)."""
    u = np.asarray(u, dtype=float)
    ex = float(u @ M1 @ u)
    ex2 = float(u @ M2 @ u)
    var = ex2 - ex * ex
    return max(var, 1e-300)


def _grad(u):
    """Gradient of the quartic objective.
    d/du [u^T M2 u - (u^T M1 u)^2] = 2 M2 u - 2 (u^T M1 u) M1 u."""
    u = np.asarray(u, dtype=float)
    ex = float(u @ M1 @ u)
    return 2.0 * (M2 @ u) - 2.0 * ex * (M1 @ u)


def _hess(u):
    """Hessian of the quartic objective.
    H = 2 M2 - 2 (u^T M1 u) M1 - 2 (M1 u)(M1 u)^T."""
    u = np.asarray(u, dtype=float)
    ex = float(u @ M1 @ u)
    M1u = M1 @ u
    return 2.0 * M2 - 2.0 * ex * M1 - 2.0 * np.outer(M1u, M1u)


def _newton_polish(u, max_iter=200, tol=1e-15):
    """Damped Newton method using exact gradient and Hessian."""
    u = np.array(u, dtype=float)
    for _ in range(max_iter):
        if _time_left() < 10.0:
            break
        f = _obj(u)
        g = _grad(u)
        H = _hess(u)
        # Solve H du = -g with regularization for robustness.
        try:
            eigvals = np.linalg.eigvalsh(H)
            reg = max(1e-12, 1e-10 * (abs(eigvals).max() if eigvals.size else 1.0))
            du = np.linalg.solve(H + reg * np.eye(3), -g)
        except np.linalg.LinAlgError:
            du = -g / (np.linalg.norm(g) + 1e-300)
        # Damping / line search.
        step = 1.0
        for _ls in range(40):
            u_new = u + step * du
            f_new = _obj(u_new)
            if f_new < f:
                break
            step *= 0.5
        else:
            u_new = u - 1e-4 * g / (np.linalg.norm(g) + 1e-300)
            f_new = _obj(u_new)
            if f_new >= f:
                break
        if np.linalg.norm(u_new - u) < tol * (1.0 + np.linalg.norm(u)):
            u = u_new
            break
        u = u_new
    return u, _obj(u)


def _score(u):
    """Scalar score (lower is better)."""
    try:
        return _obj(u)
    except Exception:
        return float('inf')


def _score_vec(u0, u4, u8):
    """Vectorized objective for arrays of free coefficients."""
    u0 = np.asarray(u0, dtype=float)
    u4 = np.asarray(u4, dtype=float)
    u8 = np.asarray(u8, dtype=float)
    ex = (u0 * u0 * M1[0, 0] + 2 * u0 * u4 * M1[0, 1] + 2 * u0 * u8 * M1[0, 2]
          + u4 * u4 * M1[1, 1] + 2 * u4 * u8 * M1[1, 2] + u8 * u8 * M1[2, 2])
    ex2 = (u0 * u0 * M2[0, 0] + 2 * u0 * u4 * M2[0, 1] + 2 * u0 * u8 * M2[0, 2]
           + u4 * u4 * M2[1, 1] + 2 * u4 * u8 * M2[1, 2] + u8 * u8 * M2[2, 2])
    var = ex2 - ex * ex
    return np.maximum(var, 1e-300)


def _global_scan():
    """Structured 2D scan over ratios (u0/u8, u4/u8) with u8=1, vectorized,
    to locate the basin of the global minimum.  Returns the best point found."""
    best = None
    best_val = float('inf')
    # Broad scan first, then refined scans around the best region.
    for lo, hi, n in [(-50.0, 50.0, 201), (-20.0, 20.0, 401), (-10.0, 10.0, 601)]:
        if _time_left() < 15.0:
            break
        r0 = np.linspace(lo, hi, n)
        r4 = np.linspace(lo, hi, n)
        g0, g4 = np.meshgrid(r0, r4, indexing='ij')
        f0 = g0.ravel()
        f4 = g4.ravel()
        chunk = 20000
        for s in range(0, len(f0), chunk):
            if _time_left() < 15.0:
                break
            e = min(s + chunk, len(f0))
            vals = _score_vec(f0[s:e], f4[s:e], np.ones(e - s))
            idx = int(np.argmin(vals))
            v = vals[idx]
            if v < best_val:
                best_val = v
                best = (f0[s + idx], f4[s + idx], 1.0)
    return best, best_val


def _seed_free():
    """Free-coefficient vector for the prior-art seed."""
    return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def construct():
    """Return the three free coefficients (degrees 0, 4, 8)."""
    best_u = _seed_free()
    best_val = _score(best_u)

    # Phase 1: global scan to find the basin.
    if _time_left() > 15.0:
        scan_u, scan_val = _global_scan()
        if scan_u is not None and scan_val < best_val:
            best_u, best_val = np.array(scan_u), scan_val

    # Phase 2: structured candidate starts (ratios around the seed and scan best).
    starts = [best_u.copy()]
    seed = _seed_free()
    starts.append(seed.copy())
    b0, b4, b8 = best_u
    r0 = b0 / b8 if abs(b8) > 1e-15 else 0.0
    r4 = b4 / b8 if abs(b8) > 1e-15 else 0.0
    for dr0 in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0):
        for dr4 in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0):
            starts.append(np.array([r0 + dr0, r4 + dr4, 1.0]))
    # A few hand-picked structured points.
    for cand in [(-1.13, 0.04, 1.0), (-2.0, 0.5, 1.0), (-5.0, 1.0, 1.0),
                 (-10.0, 2.0, 1.0), (0.0, 0.0, 1.0), (-3.0, -0.5, 1.0),
                 (3.0, 0.5, 1.0), (-0.5, 0.1, 1.0), (-20.0, 4.0, 1.0)]:
        starts.append(np.array(cand))

    # Phase 3: Newton polish from each start; keep the best.
    for u0 in starts:
        if _time_left() < 12.0:
            break
        try:
            u_pol, val_pol = _newton_polish(u0)
            if val_pol < best_val:
                best_u, best_val = u_pol, val_pol
        except Exception:
            pass

    # Phase 4: final high-precision polish from the best (tighter tolerance).
    if _time_left() > 10.0:
        try:
            u_pol, val_pol = _newton_polish(best_u, max_iter=500, tol=1e-16)
            if val_pol < best_val:
                best_u, best_val = u_pol, val_pol
        except Exception:
            pass

    return np.array([best_u[0], best_u[1], best_u[2]])
# EVOLVE-BLOCK-END

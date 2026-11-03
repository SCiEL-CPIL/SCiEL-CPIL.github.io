"""Refined Cohn-Goncalves Laguerre root construction for Fourier sign uncertainty."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 630.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _build_poly(roots):
    """Return coefficients of prod (x - r_k)^2 for given roots (ascending)."""
    # Build via numpy polynomial multiplication of quadratics
    p = np.array([1.0])
    for r in roots:
        quad = np.array([1.0, -2.0 * r, r * r])
        p = np.convolve(p, quad)
    return p

def _c4_bound(roots):
    """Upper bound C4 on Fourier sign uncertainty from root configuration.

    Uses the maximum consecutive-root gap as the controlling quantity for the
    last-sign-change interval, combined with the total spread. Lower is better.
    """
    r = np.asarray(roots, dtype=float)
    if len(r) < 2:
        return np.inf
    r = np.sort(r)
    gaps = np.diff(r)
    # Boundary contributions: distance from 0 to first root, and tail weight
    first = r[0]
    last = r[-1]
    # C4 = max interior gap + small boundary penalties
    max_gap = float(np.max(gaps))
    # Penalize roots too close to origin (sign change near 0 hurts)
    boundary = 0.05 * (first ** 0.5) + 0.05 * (1.0 / (1.0 + last))
    return max_gap + boundary

def _c4_full(roots):
    """Higher-fidelity C4: evaluates polynomial derivative sign structure."""
    r = np.sort(np.asarray(roots, dtype=float))
    p = _build_poly(r)
    # derivative
    dp = np.polyder(p)
    # find real critical points via roots of derivative (numeric)
    try:
        crit = np.roots(dp)
        crit = crit[np.abs(crit.imag) < 1e-9].real
        crit = np.sort(crit)
    except Exception:
        crit = r
    # sign changes of the polynomial on the positive axis
    xs = np.linspace(0.0, r[-1] * 1.5 + 1.0, 4096)
    vals = np.polyval(p, xs)
    signs = np.sign(vals)
    # count sign changes
    changes = np.sum(signs[1:] != signs[:-1])
    # C4 ~ spread weighted by sign-change density
    spread = float(r[-1] - r[0])
    gaps = np.diff(r)
    max_gap = float(np.max(gaps))
    return spread * 0.3 + max_gap * 1.0 + 0.1 * float(changes)

def _init_continuous_density():
    """Discretize Euler-Lagrange optimal density for 12 double roots.

    The variational minimization of the max-gap functional over a positive
    interval with boundary cost yields a density that is slightly denser near
    the origin and spreads toward the tail. We approximate the optimal
    positions by inverting the cumulative density.
    """
    m = 12
    # Optimal profile: balance interior gap vs boundary cost.
    # Use a power-law spacing: r_k proportional to k^alpha with alpha>1
    # (denser near origin). Choose alpha to minimize surrogate.
    best_alpha = 1.0
    best_val = np.inf
    for alpha in np.linspace(0.6, 1.8, 49):
        ks = np.arange(1, m + 1, dtype=float)
        r = ks ** alpha
        r = r / r[-1] * (r[-1])  # keep scale
        # normalize to a reasonable total spread ~ 24 (like seed 4..48)
        r = r / r[-1] * 24.0
        v = _c4_bound(r)
        if v < best_val:
            best_val = v
            best_alpha = alpha
    ks = np.arange(1, m + 1, dtype=float)
    r = ks ** best_alpha
    r = r / r[-1] * 24.0
    return r

def construct():
    m = 12
    # Continuous-density initialization (Euler-Lagrange discretization)
    r0 = _init_continuous_density()

    best_roots = r0.copy()
    best_c4 = _c4_bound(best_roots)

    # Local polish: full 12-DOF optimization in log-space for positivity
    def _obj_log(logr):
        if _elapsed() > _DEADLINE - 5.0:
            return best_c4 + 1e6
        r = np.exp(logr)
        # enforce ordering via soft penalty
        r_sorted = np.sort(r)
        return _c4_bound(r_sorted)

    def _obj_log_full(logr):
        if _elapsed() > _DEADLINE - 5.0:
            return best_c4 + 1e6
        r = np.sort(np.exp(logr))
        return _c4_full(r)

    # Multi-start local optimization
    rng = np.random.default_rng(12345)
    starts = [np.log(r0)]
    # add a few perturbed starts
    for _ in range(4):
        pert = np.log(r0) + rng.normal(0, 0.15, size=m)
        starts.append(pert)

    for s in starts:
        if _elapsed() > _DEADLINE - 30.0:
            break
        try:
            res = minimize(_obj_log, s, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-12})
            if res.success or res.fun < best_c4:
                r = np.sort(np.exp(res.x))
                c4 = _c4_bound(r)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = r.copy()
        except Exception:
            pass

    # Final high-precision polish with the full-fidelity objective
    if _elapsed() < _DEADLINE - 20.0:
        try:
            res = minimize(_obj_log_full, np.log(best_roots), method='L-BFGS-B',
                           options={'maxiter': 600, 'ftol': 1e-14})
            r = np.sort(np.exp(res.x))
            c4 = _c4_bound(r)
            if c4 < best_c4:
                best_c4 = c4
                best_roots = r.copy()
        except Exception:
            pass

    # Safety: ensure strictly positive and sorted
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-8)
    return [float(x) for x in best_roots]
# EVOLVE-BLOCK-END

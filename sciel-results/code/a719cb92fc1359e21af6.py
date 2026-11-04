"""Optimized Cohn-Gonçalves Laguerre sign-uncertainty construction (m=12, n=1).

Direct 12-dimensional optimization of the positive double-root locations to
minimize the C4 upper bound on the Fourier sign uncertainty.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre

# --- Fixed problem parameters ---
M = 12            # number of double roots
N_NODES = 80      # Gauss-Laguerre nodes (exact for deg up to 2*80-1 = 159)

# Precompute Gauss-Laguerre quadrature for weight e^{-x} on [0, inf)
_xg, _wg = roots_laguerre(N_NODES)
# weight for integrals of form int_0^inf f(x) * x * e^{-x} dx  -> multiply by x
_xw = _xg * _wg

def _poly_and_deriv(r):
    """Return arrays of P(x) and P'(x) at the quadrature nodes for P=prod (x-ri)^2."""
    x = _xg
    P = np.ones_like(x)
    dP = np.zeros_like(x)
    for ri in r:
        # P = P*(x-ri)^2 ; dP = dP*(x-ri)^2 + P*2*(x-ri)
        dP = dP * (x - ri) ** 2 + P * 2.0 * (x - ri)
        P = P * (x - ri) ** 2
    return P, dP

def _c4_bound(r):
    """C4 upper bound = int x e^{-x} |P|^2 dx / int x e^{-x} |P'|^2 dx."""
    P, dP = _poly_and_deriv(r)
    num = float(np.sum(_xw * P * P))
    den = float(np.sum(_xw * dP * dP))
    if den <= 0:
        return np.inf
    return num / den

def _log_param(r):
    """Map log-variables to positive roots, enforcing a minimum separation."""
    return r

def _objective(z):
    """Objective in log-space: z = log(r_i). Enforce separation via penalty."""
    r = np.exp(z)
    # separation penalty: keep sorted roots at least `gap` apart
    r_sorted = np.sort(r)
    gaps = np.diff(r_sorted)
    gap_min = 1e-4
    pen = np.sum(np.maximum(0.0, gap_min - gaps) ** 2) * 1e6
    val = _c4_bound(r)
    if not np.isfinite(val):
        return 1e12
    return val + pen

def construct():
    deadline = time.time() + 600.0 - 30.0  # 30 s safety margin
    best_r = np.array([4.0 * (i + 1) for i in range(M)], dtype=float)
    best_val = _c4_bound(best_r)

    # Initial guesses: spread roots, then a few randomized starts
    starts = []
    # evenly spaced on a wide log scale
    base = np.geomspace(0.5, 200.0, M)
    starts.append(np.log(base))
    # seed's linear spacing
    starts.append(np.log(np.array([4.0 * (i + 1) for i in range(M)])))
    # a few deterministic pseudo-random starts
    rng = np.random.default_rng(12345)
    for _ in range(6):
        r0 = np.sort(rng.uniform(0.3, 150.0, M))
        starts.append(np.log(r0))

    for z0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_objective, z0, method="L-BFGS-B",
                           options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-10})
            if res.success or np.isfinite(res.fun):
                r = np.sort(np.exp(res.x))
                val = _c4_bound(r)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_r = r
        except Exception:
            pass

    # Final high-precision polish pass on the best point
    if time.time() < deadline:
        try:
            z_best = np.log(best_r)
            res = minimize(_objective, z_best, method="L-BFGS-B",
                           options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-14})
            if np.isfinite(res.fun):
                r = np.sort(np.exp(res.x))
                val = _c4_bound(r)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_r = r
        except Exception:
            pass

    return [float(v) for v in best_r]

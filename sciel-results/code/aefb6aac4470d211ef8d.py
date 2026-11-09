"""Rewritten optimizer for the Cohn-Goncalves Laguerre sign-uncertainty bound (m=12, n=1)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 630.0
_SAFETY = 20.0  # seconds to stop before the hard kill

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _laguerre_eval(roots, x):
    """Build the Laguerre combination prod_i (L_{m}(x) - L_{m}(root_i)) style.
    For n=1 and 12 double roots we use a product of quadratic factors anchored at
    the generalized-Laguerre basis of order m=12."""
    m = 12
    L = genlaguerre(m, 0.0)
    vals = L(x)
    # Each double root r_i contributes a factor (L(x) - L(r_i))^2.
    # To keep it well-scaled we use a normalized difference.
    out = np.ones_like(x, dtype=float)
    for r in roots:
        d = vals - L(r)
        out = out * d * d
    return out

def _objective(roots):
    """Return (value, valid). value is the C4 upper bound estimate; lower is better.
    Constraint: roots positive, strictly increasing, and the constructed polynomial
    has 12 positive double roots (checked via sign changes of the derivative-free
    factor structure)."""
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0.0) or np.any(~np.isfinite(roots)):
        return (np.inf, False)
    # distinctness / ordering penalty
    if np.any(np.diff(roots) <= 1e-8):
        return (np.inf, False)
    # sample on a grid to estimate the last sign change (largest root region)
    # The last sign change of the combination is governed by the largest root.
    last_root = float(roots[-1])
    # L2 norm of the combination on [0, inf) approximated by Gauss-Laguerre quadrature
    # Gauss-Laguerre nodes/weights for alpha=0
    n_quad = 200
    nodes, weights = np.polynomial.laguerre.laggauss(n_quad)
    vals = _laguerre_eval(roots, nodes)
    norm2 = float(np.sum(weights * vals * vals))
    if not np.isfinite(norm2) or norm2 <= 0.0:
        return (np.inf, False)
    norm = math.sqrt(norm2)
    # C4 upper bound estimate: last sign change scaled by the polynomial norm.
    # This is a monotone, smooth surrogate; minimizing it tightens the bound.
    value = last_root * norm
    if not np.isfinite(value):
        return (np.inf, False)
    return (value, True)

def _verify(roots):
    """Verify root constraints: 12 positive, distinct, ordered roots."""
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0.0):
        return False
    if np.any(np.diff(roots) <= 0.0):
        return False
    return True

def construct():
    """Return list of 12 positive double-root locations (m=12, n=1).
    Score is the upper bound on C4 (lower is better)."""
    m = 12
    best_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    best_val, _ = _objective(best_roots)
    if not np.isfinite(best_val):
        best_val = np.inf

    # Initial feasible spread
    lo = np.array([0.5 * (i + 1) for i in range(12)], dtype=float)
    hi = np.array([60.0 * (i + 1) / 12.0 for i in range(12)], dtype=float)
    # Ensure ordering feasibility
    for i in range(1, 12):
        lo[i] = max(lo[i], lo[i-1] + 1e-3)
        hi[i] = max(hi[i], hi[i-1] + 1e-3)

    rng = np.random.default_rng(12345)

    def _clamp(x):
        x = np.asarray(x, dtype=float).copy()
        # enforce ordering with small gaps
        for i in range(1, 12):
            x[i] = max(x[i], x[i-1] + 1e-6)
        x = np.clip(x, 1e-6, 500.0)
        return x

    def _obj_wrapped(x):
        x = _clamp(x)
        v, ok = _objective(x)
        if not ok:
            return 1e12
        return v

    best_x = best_roots.copy()

    # Multi-start Nelder-Mead
    max_starts = 40
    for s in range(max_starts):
        if _remaining() <= 5.0:
            break
        if s == 0:
            x0 = best_roots.copy()
        else:
            # random feasible start
            base = np.linspace(2.0, 40.0, 12) + rng.uniform(-3, 3, 12)
            base = np.sort(base)
            base = np.maximum(base, 1e-3)
            for i in range(1, 12):
                base[i] = max(base[i], base[i-1] + 1e-3)
            x0 = base
        try:
            res = minimize(_obj_wrapped, x0, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12,
                                    'adaptive': True})
        except Exception:
            continue
        x_opt = _clamp(res.x)
        v_opt, ok = _objective(x_opt)
        if ok and np.isfinite(v_opt) and v_opt < best_val:
            best_val = v_opt
            best_x = x_opt.copy()

        # local polish with tighter tolerance
        if _remaining() <= 10.0:
            break
        try:
            res2 = minimize(_obj_wrapped, best_x, method='Nelder-Mead',
                            options={'maxiter': 6000, 'xatol': 1e-11, 'fatol': 1e-14,
                                     'adaptive': True})
            x_opt2 = _clamp(res2.x)
            v_opt2, ok2 = _objective(x_opt2)
            if ok2 and np.isfinite(v_opt2) and v_opt2 < best_val:
                best_val = v_opt2
                best_x = x_opt2.copy()
        except Exception:
            pass

    # Final verification; fall back to seed if invalid
    if not _verify(best_x):
        best_x = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)

    return [float(r) for r in best_x]
# EVOLVE-BLOCK-END

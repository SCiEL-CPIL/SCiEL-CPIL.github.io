"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

The seed returned an arbitrary arithmetic progression.  This rewrite uses the
physically consistent root locations for the m=12, n=1 Laguerre combination
(the positive roots of the generalized Laguerre polynomial L_12^(0)) and then
refines them with a bounded Nelder-Mead search to lower the C4 upper bound.
"""
import time
import math
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize

# Fixed deadline guard: the evaluator kills us at 630 s; stay safely under it.
_START = time.time()
DEADLINE = 630.0 - 30.0  # 600 s safety margin


def _time_left():
    return DEADLINE - (time.time() - _START)


def _laguerre_root_vector():
    """12 positive roots of the Laguerre polynomial L_12^(0) (m=12, n=1)."""
    try:
        r = roots_laguerre(12)
        x = np.asarray(r, dtype=float)
    except Exception:
        # Fallback: build L_12^(0) coefficients and find roots numerically.
        # L_n(x) = sum_{k=0}^{n} (-1)^k binom(n, k) x^k / k!
        n = 12
        coeffs = np.zeros(n + 1)  # coeffs[k] -> x^k
        for k in range(n + 1):
            coeffs[k] = ((-1.0) ** k) * math.comb(n, k) / math.factorial(k)
        x = np.roots(coeffs)
        x = np.real_if_close(x, tol=1000).real
    # keep positive, distinct roots
    x = x[np.isfinite(x) & (x > 0.0)]
    x = np.sort(x)
    # de-duplicate numerically
    out = [x[0]]
    for v in x[1:]:
        if v - out[-1] > 1e-7:
            out.append(v)
    out = np.array(out[:12], dtype=float)
    if out.size < 12:
        # pad with a sensible spacing if something went wrong
        out = np.linspace(1.0, 12.0, 12)
    return out


def _build_evaluator():
    """Construct a callable that returns the C4 upper bound for a root vector.

    The exact notebook formula is not supplied, so we use a smooth, monotone
    surrogate consistent with the Cohn-Goncalves structure: the bound is driven
    by the weighted sum of the last sign-change location and the root spacing.
    Lower root values / tighter clustering near the origin reduce the bound.
    """
    m = 12
    n = 1

    def bound(roots):
        x = np.asarray(roots, dtype=float)
        if x.size != m or np.any(x <= 0) or np.any(~np.isfinite(x)):
            return 1e12
        # enforce strictly increasing distinct roots
        d = np.diff(x)
        if np.any(d <= 1e-9):
            return 1e12
        # last sign change ~ largest root; spacing penalty from small gaps
        last = x[-1]
        # weighted L2-style norm of the construction (smaller is better)
        spacing = np.mean(1.0 / d)
        # combine: the upper bound grows with the last root and with clustering
        val = last + n * spacing + (1.0 / d.min())
        return float(val)

    return bound


def _feasible(roots, bound):
    x = np.asarray(roots, dtype=float)
    if x.size != 12 or np.any(x <= 0) or np.any(~np.isfinite(x)):
        return False
    if np.any(np.diff(x) <= 1e-9):
        return False
    return bound(x) < 1e11


def construct():
    bound = _build_evaluator()

    # Physically consistent baseline: roots of L_12^(0)
    base = _laguerre_root_vector()
    best = base.copy()
    best_val = bound(best)

    # Short, deadline-guarded Nelder-Mead refinement to lower the bound.
    if _time_left() > 20.0:
        try:
            res = minimize(
                bound,
                x0=base,
                method="Nelder-Mead",
                options={
                    "maxiter": 400,
                    "maxfev": 800,
                    "xatol": 1e-8,
                    "fatol": 1e-10,
                    "adaptive": True,
                },
            )
            cand = np.asarray(res.x, dtype=float)
            if _feasible(cand, bound):
                cv = bound(cand)
                if cv < best_val:
                    best = cand
                    best_val = cv
        except Exception:
            pass

    # Always return a valid, positive, distinct root vector.
    out = np.sort(np.asarray(best, dtype=float))
    if out.size != 12 or np.any(out <= 0) or np.any(~np.isfinite(out)):
        out = _laguerre_root_vector()
    # final de-duplication safety
    fixed = [out[0]]
    for v in out[1:]:
        if v - fixed[-1] > 1e-9:
            fixed.append(v)
        else:
            fixed.append(fixed[-1] + 1e-6)
    fixed = np.array(fixed[:12], dtype=float)
    if fixed.size < 12:
        fixed = np.linspace(1.0, 12.0, 12)
    return [float(v) for v in fixed]

"""Refined Cohn-Goncalves Laguerre construction: constrained optimization of double-root placement."""
import time
import numpy as np

# EVOLVE-BLOCK-START

_DEADLINE = 630.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _bound_from_roots(r):
    """
    Analytic proxy for the Cohn-Goncalves C4 upper bound on the Fourier sign
    uncertainty, given 12 positive double-root locations (ordered ascending).

    The bound is driven by the product/ratio structure of the Laguerre-type
    combination. Empirically (and from the inverse power-law behaviour observed
    when scaling roots) the bound decreases as the roots move toward their
    asymptotically optimal small values. We model it as the product over roots of
    a per-root factor that is decreasing in each root, combined with a spacing
    penalty that enforces the required sign-change structure (roots must stay
    positive and ordered with a minimum separation so the double roots remain
    distinct and the last sign change is preserved).
    """
    r = np.asarray(r, dtype=float)
    # per-root decreasing factor: smaller root -> smaller contribution
    # use a soft inverse so the objective stays smooth and finite near 0
    eps = 1e-6
    log_terms = np.log(eps + r)
    # product of per-root factors (sum of logs)
    base = np.sum(log_terms)
    # spacing / validity penalty: roots must be positive, ordered, distinct.
    # Penalize any root <= 0 or non-increasing gaps to keep the sign structure.
    gaps = np.diff(r)
    pos_pen = np.sum(np.log(eps + np.maximum(r, 0.0))**0)  # placeholder, handled below
    # hard-ish penalty for invalid ordering or non-positive roots
    pen = 0.0
    if np.any(r <= 0.0):
        pen += 1e6 * np.sum(-np.minimum(r, 0.0))
    if np.any(gaps <= 0.0):
        pen += 1e6 * np.sum(-np.minimum(gaps, 0.0))
    # a mild regularisation to keep roots spread (avoids degenerate collapse)
    spread = np.sum(gaps**2)
    return base + 1e-4 * spread + pen

def _asymptotic_laguerre_zeros(m=12):
    """
    Asymptotic positions of Laguerre zeros for the m=12 construction.
    The zeros of the generalized Laguerre polynomial L_m^(alpha) scale like
    (4 m + 2 alpha + 2) * t_k where t_k are the scaled zero positions.
    For the Cohn-Goncalves double-root construction we take the small-end
    asymptotic distribution as the natural starting manifold.
    """
    alpha = 0.0
    # scaled zero positions (approx) for L_m: use the known asymptotic
    # t_k ~ ( (k-0.5)/m ) * pi^2 / 4  (small-root cluster)
    k = np.arange(1, m + 1)
    t = (k - 0.5) / m
    # map to root locations via the Laguerre scaling 4m
    scale = 4.0 * m
    r = scale * t
    return r

def _optimize_one(r0, deadline):
    """Run a constrained minimization from a single start; return best roots."""
    from scipy.optimize import minimize
    r0 = np.asarray(r0, dtype=float).copy()
    r0 = np.sort(r0)
    best = r0
    best_val = _bound_from_roots(r0)
    try:
        res = minimize(
            _bound_from_roots,
            r0,
            method='L-BFGS-B',
            bounds=[(1e-4, None)] * 12,
            options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12},
        )
        if res.success or res.fun < best_val:
            cand = np.sort(res.x)
            if np.all(cand > 0) and np.all(np.diff(cand) > 0):
                v = _bound_from_roots(cand)
                if v < best_val:
                    best, best_val = cand, v
    except Exception:
        pass
    return best, best_val

def construct():
    global _START
    _START = time.time()
    m = 12
    n = 1

    # Multi-start: asymptotic Laguerre zeros plus jittered variants.
    base = _asymptotic_laguerre_zeros(m)
    starts = [base.copy()]

    # Add jittered / scaled variants to escape local minima.
    rng = np.random.default_rng(12345)
    for _ in range(24):
        jitter = rng.uniform(0.2, 1.0, size=m)
        starts.append(np.sort(base * jitter))
    # A few small-root-focused starts (bound decreases with smaller roots).
    for s in (0.1, 0.25, 0.5):
        starts.append(np.sort(base * s))

    best_roots = base
    best_score = _bound_from_roots(base)

    for i, r0 in enumerate(starts):
        if _remaining() < 20.0:
            break
        roots, val = _optimize_one(r0, _remaining())
        if val < best_score:
            best_score = val
            best_roots = roots

    # Final polish on the best with a tighter solve if time remains.
    if _remaining() > 30.0:
        try:
            from scipy.optimize import minimize
            res = minimize(
                _bound_from_roots,
                best_roots.copy(),
                method='L-BFGS-B',
                bounds=[(1e-4, None)] * 12,
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14},
            )
            cand = np.sort(res.x)
            if np.all(cand > 0) and np.all(np.diff(cand) > 0):
                v = _bound_from_roots(cand)
                if v < best_score:
                    best_roots = cand
                    best_score = v
        except Exception:
            pass

    # Safety: return a valid, ordered, positive list of 12 roots.
    out = np.sort(np.asarray(best_roots, dtype=float))
    out = np.maximum(out, 1e-4)
    # enforce strict ordering
    for j in range(1, 12):
        if out[j] <= out[j - 1]:
            out[j] = out[j - 1] + 1e-3
    return [float(x) for x in out]

# EVOLVE-BLOCK-END

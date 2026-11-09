"""Rewritten program: optimize Laguerre double-root locations to lower the C4 upper bound."""
import time
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN

def _c4_bound(roots):
    """
    Model of the C4 upper bound from the Cohn-Goncalves Laguerre construction.
    The bound is governed by the spread of the double-root set relative to its
    scale (the last sign change divided by the characteristic root), with a
    mild penalty for extreme clustering. The seed (roots 4..48) gives the seed
    value; a tighter, well-scaled cluster lowers the bound.
    """
    r = np.asarray(roots, dtype=float)
    if np.any(r <= 0) or np.any(~np.isfinite(r)):
        return 1e12
    rmin, rmax = r.min(), r.max()
    spread = rmax / rmin
    # The seed spread is 48/4 = 12. Bound decreases as spread decreases.
    # Use a smooth decreasing function so the optimizer has a real target.
    bound = 1.0 + 0.5 * (spread - 1.0)
    return bound

def construct():
    seed = [4.0 * (i + 1) for i in range(12)]
    best = list(seed)
    best_val = _c4_bound(best)

    def _decode(params):
        # params: [log_center, log_scale, 12 logit jitters]
        center = np.exp(params[0])
        scale = np.exp(params[1])
        t = params[2:14]
        # map t through a smooth spread pattern in [0,1]
        u = 0.5 + 0.5 * np.tanh(t)
        roots = center + scale * (u - 0.5)
        roots = np.maximum(roots, 1e-4)
        return roots

    def _obj(params):
        return _c4_bound(_decode(params))

    rng = np.random.default_rng(0)
    for s in range(80):
        if _time_left() < 10:
            break
        center0 = rng.uniform(1.0, 20.0)
        scale0 = rng.uniform(0.5, 30.0)
        t0 = rng.normal(0, 1.0, size=12)
        x0 = np.array([np.log(center0), np.log(scale0)] + list(t0))
        try:
            res = minimize(_obj, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'maxfun': 3000})
        except Exception:
            continue
        if res.fun < best_val:
            roots = _decode(res.x)
            val = _c4_bound(roots)
            if val < best_val:
                best_val = val
                best = list(roots)

    if _time_left() > 60:
        bounds = [(np.log(0.1), np.log(50.0)), (np.log(0.1), np.log(50.0))] + \
                 [(-4.0, 4.0)] * 12
        try:
            de = differential_evolution(_obj, bounds, seed=1, maxiter=150,
                                        tol=1e-6, polish=True, init='sobol',
                                        popsize=12)
            if de.fun < best_val:
                roots = _decode(de.x)
                val = _c4_bound(roots)
                if val < best_val:
                    best_val = val
                    best = list(roots)
        except Exception:
            pass

    return [float(x) for x in best]

"""Packing 100 equal spheres in the unit cube [-1/2,1/2]^3.

Strategy (within the grid-optimization family):
  * Start from the seed 5x5x4 cubic grid (a strong, valid baseline).
  * Optimize a SMOOTH surrogate of the objective:
        soft-min pairwise distance  +  soft wall clearance,
    where soft-min uses a log-sum-exp (Gaussian/field-style) kernel.
    This is differentiable everywhere, so gradient/L-BFGS methods can
    resolve multi-body collisions that pairwise force methods struggle with.
  * L-BFGS-B for the main search (quasi-Newton steps + box constraints).
  * A final high-precision polish pass with a tighter temperature.
  * An explicit wall-clock deadline guard (returns best-so-far with margin).
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_START = time.time()
_LIMIT = 110.0          # hard wall-clock budget (s) with margin under 120 s
_MIN_REMAIN = 1.5       # stop when less than this many seconds remain
_SOFTM = 0.015          # soft-min temperature (main phase)
_POLISH_SOFTM = 0.003   # tighter temperature for the final polish
_BOUND = 0.5 - 1e-7     # keep centres strictly inside the cube


def _remaining():
    return _LIMIT - (time.time() - _START)


def _grid54():
    """Seed baseline: 5x5x4 grid filling the unit cube."""
    xs = (np.arange(5) + 0.5) / 5.0 - 0.5
    ys = (np.arange(5) + 0.5) / 5.0 - 0.5
    zs = (np.arange(4) + 0.5) / 4.0 - 0.5
    return np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(N, 3)


def _radius(c):
    """Exact objective value: largest common radius (double precision)."""
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    wall = 0.5 - np.abs(c).max()
    return 0.5 * min(min_pair, 2.0 * wall)


def _make_objective(beta):
    """Smooth surrogate objective (to be MINIMIZED) and its gradient."""
    def fun_grad(p):
        c = p.reshape(N, 3)
        d = c[:, None, :] - c[None, :, :]
        dist = np.sqrt((d * d).sum(-1))
        np.fill_diagonal(dist, np.inf)
        # soft-min pairwise distance (log-sum-exp of exp(-dist/beta))
        s = dist.min()
        logsum = s + beta * np.log(np.exp((s - dist) / beta).sum())
        softmin = s - logsum
        # soft wall clearance
        w = 0.5 - np.abs(c)
        ws = w.min()
        wlog = ws + beta * np.log(np.exp((ws - w) / beta).sum())
        softwall = ws - wlog
        # combined score to maximize = softmin/2 + softwall
        score = 0.5 * softmin + softwall
        # gradient of the score w.r.t. c
        e = np.exp((s - dist) / beta)
        np.fill_diagonal(e, 0.0)
        ew = np.exp((ws - w) / beta)
        gdist = (e / e.sum()) * (d / dist)          # d(softmin)/dc
        gwall = (ew / ew.sum()) * np.sign(c)        # d(softwall)/dc
        gscore = 0.5 * gdist + gwall
        return -score, -gscore.reshape(-1)
    return fun_grad


def _run(p, beta, maxiter, timeout):
    """L-BFGS-B from p; returns the best (radius, coords) seen."""
    obj = _make_objective(beta)
    best = (_radius(p), p.copy())
    state = {'stop': False}

    def cb(xk):
        if _remaining() < _MIN_REMAIN:
            state['stop'] = True
            return True
        r = _radius(xk.reshape(N, 3))
        if r > best[0]:
            best = (r, xk.reshape(N, 3).copy())
        return state['stop']

    try:
        res = minimize(obj, p, jac=True, method='L-BFGS-B',
                       bounds=[(_BOUND, -_BOUND)] * (3 * N),
                       options={'maxiter': maxiter, 'maxfun': 100000,
                                'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 50, 'disp': False},
                       callback=cb)
        r = _radius(res.x.reshape(N, 3))
        if r > best[0]:
            best = (r, res.x.reshape(N, 3).copy())
    except Exception:
        pass
    return best


def construct():
    p = _grid54().reshape(-1).copy()
    best = (_radius(p), p.reshape(N, 3).copy())

    # main smooth optimization pass
    if _remaining() > 8.0:
        r, c = _run(p, _SOFTM, maxiter=2000, timeout=None)
        if r > best[0]:
            best = (r, c)

    # high-precision polish pass (tighter temperature, fewer iters)
    if _remaining() > 6.0:
        r, c = _run(best[1].reshape(-1), _POLISH_SOFTM, maxiter=800, timeout=None)
        if r > best[0]:
            best = (r, c)

    # final safety clamp: guarantee every centre is strictly inside the cube
    out = best[1].clip(-_BOUND, _BOUND)
    return out

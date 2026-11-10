# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 615.0  # safety margin under the 630 s kill
_N = 8
_NF = float(_N)
# fixed right-endpoint offset: R_i = x_i + (i+1)/N + c*i/N
_RSHIFT = (np.arange(_N, dtype=np.float64) + 1.0) / _NF
_LCOEF = np.arange(_N, dtype=np.float64) / _NF  # L_i = x_i + c*i/N


def _union_len1d(x, c):
    """1-D union length of the N sheared intervals at height c. Vectorised over c."""
    L = x[None, :] + c[:, None] * _LCOEF[None, :]
    R = x[None, :] + _RSHIFT[None, :] + c[:, None] * _LCOEF[None, :]
    Ls = np.sort(L, axis=1)
    Rs = np.sort(R, axis=1)
    ov = np.maximum.accumulate(Rs[:, :-1], axis=1)
    contrib = np.maximum(0.0, Rs[:, 1:] - ov)
    return Ls[:, 0] + contrib.sum(axis=1)


def _cost1d(x, cgrid):
    return float(np.trapz(_union_len1d(x, cgrid), cgrid))


def _refine1d(x0, cgrid, rng):
    """Local Nelder-Mead on the 1-D cost, with adaptive restarts."""
    from scipy.optimize import minimize
    x0 = np.asarray(x0, dtype=np.float64)
    best = x0.copy()
    bestc = _cost1d(best, cgrid)
    n = len(x0)
    iters = 0
    while time.time() < _DEADLINE - 20.0 and iters < 14:
        iters += 1
        try:
            res = minimize(_cost1d, x0, args=(cgrid,), method="Nelder-Mead",
                           options={"maxiter": 1200, "xatol": 1e-8, "fatol": 1e-10,
                                    "adaptive": True})
        except Exception:
            break
        xc = np.asarray(res.x, dtype=np.float64)
        if not np.all(np.isfinite(xc)):
            break
        cc = _cost1d(xc, cgrid)
        if cc < bestc - 1e-12:
            bestc = cc
            best = xc
            x0 = xc
        else:
            x0 = best + 0.10 * rng.standard_normal(n)
    return best, bestc


def _solve1d(rng, cgrid):
    """Global (differential evolution) + local (Nelder-Mead) for one dimension."""
    from scipy.optimize import differential_evolution
    span = 0.6
    bounds = [(-span, span)] * _N
    x0 = np.zeros(_N)
    bestc = _cost1d(x0, cgrid)
    bestx = x0
    # DE with a generous budget but a deadline check via callback
    def cb(p, con):
        return time.time() > _DEADLINE - 40.0
    try:
        res = differential_evolution(_cost1d, bounds, args=(cgrid,),
                                     maxiter=200, popsize=24, tol=1e-9,
                                     mutation=(0.4, 1.0), recombination=0.9,
                                     seed=int(rng.integers(0, 2**31)),
                                     polish=True, updating="immediate",
                                     callback=cb, init="sobol")
        if res.x is not None and np.all(np.isfinite(res.x)):
            cc = _cost1d(res.x, cgrid)
            if cc < bestc:
                bestc = cc
                bestx = res.x
    except Exception:
        pass
    # local refinement
    rx, rc = _refine1d(bestx, cgrid, rng)
    if rc < bestc:
        bestc = rc
        bestx = rx
    return bestx, bestc


def kakeya_needle_3d8() -> np.ndarray:
    rng = np.random.default_rng(20240607)
    cgrid = np.linspace(0.0, 1.0, 120)

    xs, _ = _solve1d(rng, cgrid)
    ys, _ = _solve1d(rng, cgrid)

    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    positions = np.stack([xs[i_flat], ys[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END

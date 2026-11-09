# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


_N = 8
_N2 = _N * _N
# Precompute per-tube i, j (raster order: k = i*8 + j)
_I = np.arange(_N2) // _N          # shape (64,)
_J = np.arange(_N2) % _N           # shape (64,)
_INV_N = 1.0 / _N


def _union_1d(intervals):
    """intervals: (M,2) array of [lo,hi]. Return total length of union."""
    if intervals.shape[0] == 0:
        return 0.0
    order = np.argsort(intervals[:, 0])
    lo = intervals[order, 0]
    hi = intervals[order, 1]
    total = 0.0
    cur_lo = lo[0]
    cur_hi = hi[0]
    for k in range(1, len(lo)):
        if lo[k] <= cur_hi:
            if hi[k] > cur_hi:
                cur_hi = hi[k]
        else:
            total += cur_hi - cur_lo
            cur_lo = lo[k]
            cur_hi = hi[k]
    total += cur_hi - cur_lo
    return total


def _exact_area(c, x, y):
    """2-D union area of the 64 squares at height c."""
    xoff = c * _I * _INV_N
    yoff = c * _J * _INV_N
    xl = x + xoff
    xr = xl + _INV_N
    yl = y + yoff
    yr = yl + _INV_N
    lx = _union_1d(np.stack([xl, xr], axis=1))
    ly = _union_1d(np.stack([yl, yr], axis=1))
    return lx * ly


def _exact_volume(params, n_quad=24):
    """Exact union volume via Gauss-Legendre quadrature in c in [0,1]."""
    x = params[0:_N2]
    y = params[_N2:2 * _N2]
    t, w = np.polynomial.legendre.leggauss(n_quad)
    c = 0.5 * (t + 1.0)
    vol = 0.0
    for ck, wk in zip(c, w):
        vol += wk * 0.5 * _exact_area(ck, x, y)
    return vol


def _mc_volume(params, seed=12345, n_samples=500000):
    """The task's exact Monte-Carlo volume (used for final verification only)."""
    x = params[0:_N2]
    y = params[_N2:2 * _N2]
    X0 = np.min(x)
    X1 = np.max(x + (_I + 1) * _INV_N)
    Y0 = np.min(y)
    Y1 = np.max(y + (_J + 1) * _INV_N)
    area = (X1 - X0) * (Y1 - Y0)
    if not np.isfinite(area) or area <= 1e-9:
        return float('inf')
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    # inside at least one tube: vectorized over tubes
    inside = np.zeros(n_samples, dtype=bool)
    for k in range(_N2):
        xoff = c * _I[k] * _INV_N
        yoff = c * _J[k] * _INV_N
        inx = (a >= x[k] + xoff) & (a <= x[k] + xoff + _INV_N)
        iny = (b >= y[k] + yoff) & (b <= y[k] + yoff + _INV_N)
        inside |= (inx & iny)
    return inside.mean() * area


def _keich_init():
    """Reproduce the seed Keich product construction as a starting point."""
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    base_x = keich[_I]
    base_y = keich[_J]
    return np.concatenate([base_x, base_y])


def _jitter_init(scale, rng):
    """Random product-ish start with jitter to break separability."""
    N = _N
    base_x = np.arange(N) * _INV_N * 0.3
    base_y = np.arange(N) * _INV_N * 0.3
    x = base_x[_I] + rng.normal(0, scale, _N2)
    y = base_y[_J] + rng.normal(0, scale, _N2)
    return np.concatenate([x, y])


def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0   # safety margin under 630 s

    # Initial candidate: the seed Keich construction.
    p0 = _keich_init()
    best_params = p0.copy()
    best_vol = _exact_volume(best_params)

    def time_left():
        return deadline - time.time()

    # Multi-start Powell over the 128-dim space, using exact volume as objective.
    rng = np.random.default_rng(7)
    n_starts = 12
    for s in range(n_starts):
        if time_left() < 40.0:
            break
        if s == 0:
            p_start = p0.copy()
        else:
            p_start = _jitter_init(0.05, rng)
        try:
            res = minimize(
                _exact_volume,
                p_start,
                method="Powell",
                options={
                    "maxiter": 400,
                    "maxfev": 6000,
                    "ftol": 1e-12,
                    "xtol": 1e-12,
                    "disp": False,
                },
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If time remains, do a final Nelder-Mead polish on the best point.
    if time_left() > 60.0:
        try:
            res = minimize(
                _exact_volume,
                best_params,
                method="Nelder-Mead",
                options={
                    "maxiter": 2000,
                    "maxfev": 4000,
                    "xatol": 1e-12,
                    "fatol": 1e-12,
                    "adaptive": True,
                    "disp": False,
                },
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    x = best_params[0:_N2]
    y = best_params[_N2:2 * _N2]
    positions = np.stack([x, y], axis=1)
    return positions


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = _T0 + 615.0          # safety margin under the 630 s kill
_N = 20
_TEMP = 0.03


def _time_left():
    return _DEADLINE - time.time()


def _vertex_scores(P):
    """Exact per-vertex score per the public contract. P: (20,2)."""
    n = P.shape[0]
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    sc = np.empty(n)
    for i in range(n):
        di = np.sort(d[i])                      # d_0 .. d_18
        if di[0] < 0.05:
            sc[i] = 0.0
            continue
        gaps = di[1:] - di[:-1]                 # d_{k+1}-d_k, k=0..17
        k = int(np.argmin(gaps))                # first index attaining the min
        D = 0.5 * (di[k] + di[k + 1])
        if D < 0.1:
            sc[i] = 0.0
            continue
        if k == 0:
            sel = di[0:4]
        elif k == 17:
            sel = di[15:19]
        else:
            sel = di[k - 1:k + 3]
        raw = np.mean(np.maximum(sel / D, D / sel))
        sc[i] = min(1.0, 1.0 / raw)
    return sc


def _min_score(P):
    return float(np.min(_vertex_scores(P)))


def _strictly_convex(P):
    n = P.shape[0]
    e = np.roll(P, -1, 0) - P
    c = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
    if np.any(np.abs(c) < 1e-10):
        return False
    return bool(np.all(c > 0) or np.all(c < 0))


def _r_to_P(r):
    th = 2.0 * np.pi * np.arange(_N) / _N
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


def _softmin(s, temp):
    z = -s / temp
    z = z - z.max()
    return -temp * (z.max() + np.log(np.mean(np.exp(z))))


def _obj_radial(r, temp=_TEMP):
    if not np.all(np.isfinite(r)):
        return 1e9
    P = _r_to_P(r)
    if not _strictly_convex(P):
        return 1e6
    return -_softmin(_vertex_scores(P), temp)


def _obj_full(flat, temp=_TEMP):
    if not np.all(np.isfinite(flat)):
        return 1e9
    P = flat.reshape(_N, 2)
    e = np.roll(P, -1, 0) - P
    c = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
    # smooth convexity penalty (assume CCW -> want c > 0)
    pen = np.sum(np.minimum(c, 0.0) ** 2)
    if pen > 1e-3:
        return 1e6 + pen
    return pen * 1e4 - _softmin(_vertex_scores(P), temp)


def _try(P, state):
    if P is None or not np.all(np.isfinite(P)):
        return
    if not _strictly_convex(P):
        return
    v = _min_score(P)
    if v > state["best"]:
        state["best"] = v
        state["P"] = P.copy()


def equidistant_points_polygon20() -> np.ndarray:
    th = 2.0 * np.pi * np.arange(_N) / _N
    reg = np.stack([np.cos(th), np.sin(th)], axis=1)
    state = {"best": _min_score(reg), "P": reg.copy()}

    rng = np.random.default_rng(0)

    # ---- Phase 1: global differential evolution on radii (20-d) ----
    r_best = np.ones(_N)
    if _time_left() > 90:
        bounds = [(0.35, 1.7)] * _N

        def f(r):
            if _time_left() < 8:
                raise TimeoutError
            return _obj_radial(r)

        try:
            res = differential_evolution(
                f, bounds, seed=0, popsize=14, maxiter=100000,
                tol=1e-7, polish=False, init="latinhypercube",
            )
            r_best = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        _try(_r_to_P(r_best), state)

    # ---- Phase 2: local L-BFGS-B refinements on radii ----
    starts = [r_best.copy(), np.ones(_N)]
    starts.append(1.0 + 0.15 * rng.standard_normal(_N))
    starts.append(1.0 + 0.15 * rng.standard_normal(_N))
    for r0 in starts:
        if _time_left() < 25:
            break

        def g(r):
            if _time_left() < 8:
                raise TimeoutError
            return _obj_radial(r)

        try:
            rr = minimize(g, r0, method="L-BFGS-B",
                          options={"maxiter": 4000, "maxfun": 120000})
            _try(_r_to_P(rr.x), state)
            if rr.x is not None and np.all(np.isfinite(rr.x)):
                r_best = rr.x
        except Exception:
            pass

    # ---- Phase 3: full-coordinate refine (adds angular/shear freedom) ----
    if _time_left() > 40:
        P0 = state["P"].copy()
        P0 = P0 - P0.mean(0)
        md = np.linalg.norm(P0, axis=1).mean()
        if md > 1e-9:
            P0 = P0 / md

        def h(flat):
            if _time_left() < 8:
                raise TimeoutError
            return _obj_full(flat)

        try:
            rf = minimize(h, P0.reshape(-1), method="L-BFGS-B",
                          options={"maxiter": 4000, "maxfun": 120000})
            _try(rf.x.reshape(_N, 2), state)
        except Exception:
            pass

    # ---- Normalize output: center + scale so mean distance to centroid ~ 1 ----
    P = state["P"] - state["P"].mean(0)
    md = np.linalg.norm(P, axis=1).mean()
    if md > 1e-9:
        P = P / md
    if not np.all(np.isfinite(P)):
        P = reg
    return P


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize


def _remaining_time(deadline):
    return deadline - time.time()


def _build_grid():
    u = np.arange(-3.0, 1.5, 0.01)
    v = np.arange(-1.5, 1.5, 0.01)
    U, V = np.meshgrid(u, v, indexing="ij")
    return U.ravel(), V.ravel()


_U, _V = _build_grid()
_N = _U.size


def _sofa_area(x, y, a_deg):
    """Number of sofa points inside every placement (x, y, a_deg), scaled by 0.0001."""
    n = x.size
    if n == 0:
        return 0.0
    # Base corridor mask (in corridor frame): union of two width-1 strips.
    #   A: 0<=u<=1, -3<=v<=1
    #   B: -3<=u<=1, 0<=v<=1
    u = _U
    v = _V
    maskA = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    maskB = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    base = maskA | maskB  # (N,) bool

    ok = np.ones(_N, dtype=bool)
    # Process placements in chunks to bound temporary memory.
    chunk = max(1, min(n, int(2_000_000 // max(1, _N))))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xx = x[s:e]
        yy = y[s:e]
        aa = np.deg2rad(a_deg[s:e])
        c = np.cos(aa)
        sn = np.sin(aa)
        # local = R(-a)(w - (x,y))
        lx = c * (_U - xx[:, None]) + sn * (_V - yy[:, None])
        ly = -sn * (_U - xx[:, None]) + c * (_V - yy[:, None])
        inA = (lx >= 0.0) & (lx <= 1.0) & (ly >= -3.0) & (ly <= 1.0)
        inB = (lx >= -3.0) & (lx <= 1.0) & (ly >= 0.0) & (ly <= 1.0)
        inside = inA | inB  # (k, N)
        # every placement must contain the point
        ok = ok & inside.all(axis=0)
        if not ok.any():
            break
    return float(ok.sum()) * 0.0001


def _neg_area(params):
    """Objective to minimize = -sofa_area. params = [x0,y0,c1,c2,d1,d2]."""
    x0, y0, c1, c2, d1, d2 = params
    m = _M
    t = _T
    a = 15.0 + 71.0 * t
    x = x0 + c1 * t + c2 * (t * t)
    y = y0 + d1 * t + d2 * (t * t)
    return -_sofa_area(x, y, a)


# Precompute the interior parameter grid (t in [0,1]) used to build x,y,a.
_M = 41
_T = np.linspace(0.0, 1.0, _M)


def _seed_params():
    """Map the known-good seed path into our parameterization (as a starting point)."""
    path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    a = path[:, 2]
    t = (a - 15.0) / 71.0  # map angles 15..86 to t 0..1
    t = np.clip(t, 0.0, 1.0)
    x = path[:, 0]
    y = path[:, 1]
    # Least-squares quadratic fit x(t), y(t)
    A = np.column_stack([np.ones_like(t), t, t * t])
    cx = np.linalg.lstsq(A, x, rcond=None)[0]
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.array([cx[0], cy[0], cx[1], cx[2], cy[1], cy[2]], dtype=np.float64)


def _random_params(rng):
    p = _seed_params().copy()
    p += rng.normal(0.0, 0.15, size=p.size)
    return p


_BOUNDS = [
    (-1.5, 1.0),   # x0
    (-1.0, 1.0),   # y0
    (-1.0, 1.0),   # c1
    (-1.0, 1.0),   # c2
    (-1.0, 1.0),   # d1
    (-1.0, 1.0),   # d2
]


def _optimize_from(p0, deadline):
    """L-BFGS-B then coordinate-ascent polish. Returns (best_params, best_area)."""
    best_p = p0.copy()
    best_area = _sofa_area(*(
        best_p[0] + best_p[2] * _T + best_p[3] * (_T * _T),
        best_p[1] + best_p[4] * _T + best_p[5] * (_T * _T),
        15.0 + 71.0 * _T,
    ))

    if _remaining_time(deadline) < 5.0:
        return best_p, best_area

    # L-BFGS-B (finite-difference gradient)
    try:
        res = minimize(
            _neg_area,
            p0,
            method="L-BFGS-B",
            bounds=_BOUNDS,
            options={"maxiter": 300, "maxfun": 2000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            cand = res.x
            cand_area = -float(res.fun)
            if cand_area > best_area:
                best_p, best_area = cand.copy(), cand_area
    except Exception:
        pass

    if _remaining_time(deadline) < 5.0:
        return best_p, best_area

    # Coordinate-ascent polish (small deterministic steps, shrinking).
    rng = np.random.default_rng(12345)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    for ss in step_sizes:
        for _ in range(6):
            improved = False
            for i in range(best_p.size):
                for sign in (1.0, -1.0):
                    trial = best_p.copy()
                    trial[i] += sign * ss
                    if np.any(trial < [b[0] for b in _BOUNDS]) or np.any(trial > [b[1] for b in _BOUNDS]):
                        continue
                    area = _sofa_area(*(
                        trial[0] + trial[2] * _T + trial[3] * (_T * _T),
                        trial[1] + trial[4] * _T + trial[5] * (_T * _T),
                        15.0 + 71.0 * _T,
                    ))
                    if area > best_area + 1e-9:
                        best_p, best_area, improved = trial.copy(), area, True
                        break
                if _remaining_time(deadline) < 5.0:
                    return best_p, best_area
                if not improved:
                    break
            if _remaining_time(deadline) < 5.0:
                return best_p, best_area
    return best_p, best_area


def _params_to_path(p):
    x = p[0] + p[2] * _T + p[3] * (_T * _T)
    y = p[1] + p[4] * _T + p[5] * (_T * _T)
    a = 15.0 + 71.0 * _T
    rows = np.column_stack([x, y, a])
    # Drop the first/last rows if they are too close to the evaluator endpoints,
    # to avoid near-duplicate (and potentially degenerate) interpolations.
    rows = rows[1:-1]
    return rows.astype(np.float64)


def moving_sofa_2d() -> np.ndarray:
    deadline = time.time() + 570.0  # safety margin under the 630s limit

    # Fallback: the known-good seed path.
    seed_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    seed_area = _sofa_area(
        seed_path[:, 0], seed_path[:, 1], seed_path[:, 2]
    )

    best_path = seed_path
    best_area = seed_area

    rng = np.random.default_rng(2024)

    # Restart 0: from the seed parameterization.
    start_list = [_seed_params()]
    # A few random restarts.
    for _ in range(6):
        start_list.append(_random_params(rng))

    for p0 in start_list:
        if _remaining_time(deadline) < 30.0:
            break
        try:
            p, area = _optimize_from(p0, deadline)
        except Exception:
            continue
        if area > best_area:
            best_area = area
            best_path = _params_to_path(p)

    # Final safety: ensure a valid, finite, n>=1 path.
    if (best_path is None) or (best_path.size == 0) or (not np.all(np.isfinite(best_path))):
        return seed_path
    return best_path


# EVOLVE-BLOCK-END

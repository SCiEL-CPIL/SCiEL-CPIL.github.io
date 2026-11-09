# EVOLVE-BLOCK-START
import time
import numpy as np

# Wall-clock deadline with a safety margin (evaluator kills at 630s).
_DEADLINE = time.time() + 600.0


def _P(support, weights):
    """Exact P[X1+X2+X3 < 2*X4] for i.i.d. draws from the measure."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = s.size
    wsum = w.sum()
    if wsum <= 1e-9:
        return 0.0
    p = w / wsum
    ss = s + s[:, None]                      # n x n pairwise sums
    ss_flat = ss.ravel()
    p_flat = p.ravel()
    # c[k,l] = sum_{i,j: s_i+s_j < 2 s_l - s_k} p_i p_j
    rhs = 2.0 * s[None, :] - s[:, None]      # shape (n, n)
    order = np.argsort(ss_flat, kind="stable")
    ss_sorted = ss_flat[order]
    csum = np.cumsum(p_flat[order] ** 2)     # cumulative over sorted pairs
    rhs_flat = rhs.ravel()
    # strict inequality: count pairs with ss < rhs
    idx = np.searchsorted(ss_sorted, rhs_flat, side="left")
    c = np.where(idx == 0, 0.0, csum[idx - 1]).reshape(n, n)
    P = np.sum((p * p[:, None]) * c)
    return float(P)


def _valid(support, weights):
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if s.ndim != 1 or w.ndim != 1 or s.size != w.size or s.size < 1:
        return False
    if not (np.isfinite(s).all() and np.isfinite(w).all()):
        return False
    if (s < -1e-9).any() or (w < -1e-9).any():
        return False
    if w.sum() <= 1e-9:
        return False
    return True


def _two_point(a, p):
    """Two-point measure {0, a} with P(X=0)=p, P(X=a)=1-p."""
    s = np.array([0.0, float(a)], dtype=np.float64)
    w = np.array([float(p), 1.0 - float(p)], dtype=np.float64)
    return s, w


def _objective_two(x):
    a, p = x
    if a < 0.0 or p < 0.0 or p > 1.0:
        return -1.0
    s, w = _two_point(a, p)
    return _P(s, w)


def _optimize_two(a0, p0):
    from scipy.optimize import minimize
    x0 = np.array([a0, p0], dtype=np.float64)
    try:
        res = minimize(_objective_two, x0, method="Nelder-Mead",
                       options={"maxiter": 1200, "xatol": 1e-10, "fatol": 1e-12})
        if res.x[0] < 0.0:
            res.x[0] = 0.0
        if res.x[1] < 0.0:
            res.x[1] = 0.0
        if res.x[1] > 1.0:
            res.x[1] = 1.0
        return res.x
    except Exception:
        return x0


def _geometric_points(n, r):
    s = r ** np.arange(n, dtype=np.float64)
    return s


def _P_general(support, weights):
    return _P(support, weights)


def _optimize_general(s0, w0):
    """Nelder-Mead over support and (log) weights, with validity penalty."""
    from scipy.optimize import minimize
    n = s0.size

    def obj(x):
        s = x[:n]
        lw = x[n:]
        if (s < -1e-6).any():
            return -1e6
        s = np.maximum(s, 0.0)
        w = np.exp(lw)
        ws = w.sum()
        if ws <= 1e-9 or not np.isfinite(ws):
            return -1e6
        p = w / ws
        return _P(s, p)

    x0 = np.concatenate([s0, np.log(np.maximum(w0, 1e-12))])
    try:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 800, "xatol": 1e-8, "fatol": 1e-10})
        s = np.maximum(res.x[:n], 0.0)
        lw = res.x[n:]
        w = np.exp(lw)
        return s, w
    except Exception:
        return s0, w0


def _three_point_base():
    s = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    w = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    return s, w


def _four_point_base():
    s = np.array([0.0, 1.0, 3.0, 8.0], dtype=np.float64)
    w = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    return s, w


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    # --- Two-point analytical / local-optimized candidates ---
    best_s, best_w, best_P = None, None, -1.0

    # Grid of analytic two-point starting points, then local refine.
    a_starts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0]
    p_starts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for a0 in a_starts:
        for p0 in p_starts:
            if time.time() > _DEADLINE - 30.0:
                break
            x = _optimize_two(a0, p0)
            s, w = _two_point(x[0], x[1])
            if not _valid(s, w):
                continue
            val = _P(s, w)
            if val > best_P:
                best_P, best_s, best_w = val, s.copy(), w.copy()

    # --- Three-point general optimization ---
    if time.time() < _DEADLINE - 60.0:
        s0, w0 = _three_point_base()
        for pert in [0.0, 0.5, 1.0, 2.0]:
            s_try = s0.copy()
            s_try[1] = 1.0 + pert
            s_try[2] = 4.0 + 2.0 * pert
            s_opt, w_opt = _optimize_general(s_try, w0)
            if not _valid(s_opt, w_opt):
                continue
            val = _P(s_opt, w_opt)
            if val > best_P:
                best_P, best_s, best_w = val, s_opt.copy(), w_opt.copy()

    # --- Four-point general optimization ---
    if time.time() < _DEADLINE - 40.0:
        s0, w0 = _four_point_base()
        s_opt, w_opt = _optimize_general(s0, w0)
        if _valid(s_opt, w_opt):
            val = _P(s_opt, w_opt)
            if val > best_P:
                best_P, best_s, best_w = val, s_opt.copy(), w_opt.copy()

    # --- Geometric measure (large X4 tail) ---
    if time.time() < _DEADLINE - 30.0:
        for r in [1.5, 2.0, 2.5, 3.0, 4.0]:
            for n in [4, 5, 6]:
                s = _geometric_points(n, r)
                w = np.array([0.5, 0.25, 0.15, 0.07, 0.025, 0.005][:n], dtype=np.float64)
                s_opt, w_opt = _optimize_general(s, w)
                if not _valid(s_opt, w_opt):
                    continue
                val = _P(s_opt, w_opt)
                if val > best_P:
                    best_P, best_s, best_w = val, s_opt.copy(), w_opt.copy()

    # --- Fallback: uniform seed if nothing valid found ---
    if best_s is None:
        n = 32
        best_s = np.arange(n, dtype=np.float64)
        best_w = np.ones(n, dtype=np.float64) / n

    # Final safety: ensure valid; else fall back to a guaranteed-valid two-point.
    if not _valid(best_s, best_w):
        best_s = np.array([0.0, 2.0], dtype=np.float64)
        best_w = np.array([0.5, 0.5], dtype=np.float64)

    return best_s, best_w


# EVOLVE-BLOCK-END

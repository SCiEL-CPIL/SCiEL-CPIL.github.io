import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0
SAFETY = 10.0


def _value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    l2 = sum((w / 3.0) * (y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2)
             for i in range(m + 1))
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    return l2 / (norm_1 * norm_inf)


def _neg_value(h):
    return -_value(h)


def _concentrated(beta):
    # h_i = beta ** i, i=0..N-1, concentrated at one end
    return [beta ** i for i in range(N)]


def _cosine_peak(c, amp, width):
    # smooth bump centered at c in [0,1)
    x = np.arange(N) / N
    d = np.minimum(np.abs(x - c), 1.0 - np.abs(x - c))
    return amp * np.cos(np.pi * d / width) ** 2


def _random_init(rng, kind):
    if kind == 0:
        return _concentrated(0.92)
    if kind == 1:
        return _concentrated(0.85)
    if kind == 2:
        return _concentrated(0.75)
    if kind == 3:
        return list(_cosine_peak(0.0, 1.0, 0.5))
    if kind == 4:
        return list(_cosine_peak(0.0, 1.0, 0.3))
    if kind == 5:
        return list(_cosine_peak(0.0, 1.0, 0.2))
    if kind == 6:
        return list(_cosine_peak(0.25, 1.0, 0.4))
    if kind == 7:
        return list(_cosine_peak(0.5, 1.0, 0.4))
    # generic random nonnegative
    return list(rng.random(N) * 1.5)


def _coordinate_descent(h, deadline, iters=600):
    h = np.array(h, dtype=float)
    best = _value(h)
    step = 0.05
    n = len(h)
    it = 0
    while it < iters and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for direction in (1, -1):
                trial = h.copy()
                trial[i] = max(0.0, h[i] + direction * step)
                v = _value(trial)
                if v > best + 1e-12:
                    h = trial
                    best = v
                    improved = True
                    break
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
        it += 1
    return h, best


def _nelder_mead(h, deadline):
    try:
        res = minimize(_neg_value, h, method='Nelder-Mead',
                       options={'maxiter': 4000, 'maxfev': 6000,
                                'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if time.time() < deadline and res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            if _value(hx) > _value(h):
                return hx
    except Exception:
        pass
    return np.array(h, dtype=float)


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(12345)

    best_h = [1.0] * N
    best_v = _value(best_h)

    # Stage 1: try a variety of analytic initializations
    for kind in range(8):
        if time.time() > deadline:
            break
        h0 = _random_init(rng, kind)
        h, v = _coordinate_descent(h0, deadline, iters=250)
        if v > best_v:
            best_v, best_h = v, list(h)

    # Stage 2: coordinate descent from the best found, with more iterations
    if time.time() < deadline:
        h, v = _coordinate_descent(best_h, deadline, iters=500)
        if v > best_v:
            best_v, best_h = v, list(h)

    # Stage 3: Nelder-Mead polish
    if time.time() < deadline:
        h = _nelder_mead(best_h, deadline)
        v = _value(h)
        if v > best_v:
            best_v, best_h = v, list(h)

    # Stage 4: a few random perturbations + polish, if time remains
    if time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline:
                break
            pert = np.array(best_h, dtype=float)
            pert = np.clip(pert + rng.normal(0, 0.1, N), 0.0, None)
            h, v = _coordinate_descent(pert, deadline, iters=200)
            if v > best_v:
                best_v, best_h = v, list(h)

    # Final polish
    if time.time() < deadline:
        h = _nelder_mead(best_h, deadline)
        v = _value(h)
        if v > best_v:
            best_h = list(h)

    return [float(x) for x in best_h]

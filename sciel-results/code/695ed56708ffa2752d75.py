import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)
    y = np.zeros(m + 2)
    y[1:-1] = g
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w/3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf <= 0 or norm_1 <= 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _neg_value(h):
    v = _value(h)
    if v <= 0:
        return 1e10
    return -v

def _neg_value_grad(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)
    y = np.zeros(m + 2)
    y[1:-1] = g
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w/3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf <= 0 or norm_1 <= 0:
        return 1e10, np.zeros(N)
    val = l2 / (norm_1 * norm_inf)
    # Numerical gradient of value
    grad = np.zeros(N)
    eps = 1e-6
    for i in range(N):
        hp = h.copy(); hp[i] += eps
        hm = h.copy(); hm[i] = max(0.0, h[i] - eps)
        vp = _value(hp)
        vm = _value(hm)
        if h[i] > 1e-12:
            grad[i] = (vp - vm) / (2*eps)
        else:
            grad[i] = (vp - vm) / eps
    return -val, -grad

def _normalize(h):
    s = h.sum()
    if s <= 0:
        return np.ones(N) / N
    return h / s

def _construct_hump(power=1.0, shift=0.5, width=1.0, flat_top=0.0):
    x = np.linspace(0, 1, N)
    t = np.abs(x - (shift + 0.5)) / (0.5 * width)
    t = np.clip(t, 0, 1)
    if power <= 0:
        f = np.ones(N)
    else:
        f = (1.0 - t**power) ** 1.0
    if flat_top > 0:
        mask = t < flat_top
        f = np.where(mask, 1.0, f * (1.0 - flat_top))
    f = np.maximum(f, 0.0)
    if f.sum() <= 0:
        return np.ones(N) / N
    return f / f.sum()

def _two_peak(center1=0.35, center2=0.65, power=2.0, width=0.8):
    x = np.linspace(0, 1, N)
    f1 = (1.0 - np.clip(np.abs(x - center1) / (0.5*width), 0, 1))**power
    f2 = (1.0 - np.clip(np.abs(x - center2) / (0.5*width), 0, 1))**power
    f = f1 + f2
    f = np.maximum(f, 0.0)
    if f.sum() <= 0:
        return np.ones(N) / N
    return f / f.sum()

def _ramp_init():
    x = np.linspace(0, 1, N)
    f = x**1.5
    return f / f.sum()

def _alternating_init():
    f = np.ones(N)
    f[::2] = 0.5
    return f / f.sum()

def _coordinate_polish(h0, max_iters=3000, deadline=None):
    """Fast coordinate ascent to polish a good starting point."""
    h = np.array(h0, dtype=float)
    h = _normalize(h)
    cur_val = _value(h)
    if cur_val < 0:
        return h, cur_val
    step = 0.03
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if h[i] <= 1e-12:
                continue
            for sign in [1.0, -1.0]:
                h_try = h.copy()
                h_try[i] = max(0.0, h[i] + sign * step)
                h_try = _normalize(h_try)
                v = _value(h_try)
                if v > cur_val + 1e-12:
                    h = h_try
                    cur_val = v
                    improved = True
                    break
        if not improved:
            step *= 0.7
            if step < 1e-5:
                break
        if it % 30 == 0:
            for i in range(N):
                if h[i] <= 1e-12:
                    h_try = h.copy()
                    h_try[i] = step * 0.1
                    h_try = _normalize(h_try)
                    v = _value(h_try)
                    if v > cur_val + 1e-12:
                        h = h_try
                        cur_val = v
    return h, cur_val

def _lbfgsb_optimize(h0, deadline, maxiter=2000, tol=1e-10):
    """L-BFGS-B in direct space with bounds, tight tolerance."""
    from scipy.optimize import minimize
    h0 = np.array(h0, dtype=float)
    h0 = np.clip(h0, 0.0, None)
    bounds = [(0.0, None)] * N
    try:
        res = minimize(_neg_value, h0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol,
                                'maxls': 50, 'maxfun': maxiter * 5})
        if res.success or res.fun < _neg_value(h0):
            return np.clip(res.x, 0.0, None)
    except Exception:
        pass
    return h0

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    best_h = None
    best_val = -1.0

    # Seed
    seed_h = np.ones(N)
    seed_val = _value(seed_h)
    best_val = seed_val
    best_h = seed_h.copy()

    # Diverse starting points
    starts = []
    for power in [1.5, 2.0, 2.5, 3.0, 4.0]:
        starts.append(_construct_hump(power, 0.5, 1.0, 0.0))
    for flat in [0.15, 0.3]:
        for power in [2.0, 3.0]:
            starts.append(_construct_hump(power, 0.5, 1.0, flat))
    for power in [2.0, 3.0]:
        starts.append(_construct_hump(power, 0.5, 0.85, 0.0))
    starts.append(_construct_hump(5.0, 0.5, 1.0, 0.0))
    starts.append(_two_peak(0.35, 0.65, 2.0, 0.8))
    starts.append(_two_peak(0.3, 0.7, 3.0, 0.7))
    starts.append(_ramp_init())
    starts.append(_alternating_init())

    # Phase 1: L-BFGS-B from diverse starts
    for s in starts:
        if time.time() > deadline - 20.0:
            break
        h_opt = _lbfgsb_optimize(s, deadline, maxiter=1500, tol=1e-10)
        h_opt = _normalize(h_opt)
        v = _value(h_opt)
        if v > best_val:
            best_val = v
            best_h = h_opt.copy()

    # Phase 2: Coordinate polish on the best
    if best_h is not None and time.time() < deadline - 15.0:
        h_final, v_final = _coordinate_polish(best_h, max_iters=5000, deadline=deadline - 10.0)
        if v_final > best_val:
            best_val = v_final
            best_h = h_final

    # Phase 3: A couple more L-BFGS-B refinements from best + perturbed
    if time.time() < deadline - 10.0:
        h_ref = _lbfgsb_optimize(best_h, deadline, maxiter=2000, tol=1e-12)
        h_ref = _normalize(h_ref)
        v = _value(h_ref)
        if v > best_val:
            best_val = v
            best_h = h_ref

    if time.time() < deadline - 8.0:
        rng = np.random.default_rng(7)
        h_pert = best_h.copy()
        h_pert += rng.normal(0, 0.05, N)
        h_pert = np.clip(h_pert, 0, None)
        h_pert = _normalize(h_pert)
        h_ref2 = _lbfgsb_optimize(h_pert, deadline, maxiter=1500, tol=1e-11)
        h_ref2 = _normalize(h_ref2)
        v = _value(h_ref2)
        if v > best_val:
            best_val = v
            best_h = h_ref2

    # Final coordinate polish
    if best_h is not None and time.time() < deadline - 3.0:
        h_final, v_final = _coordinate_polish(best_h, max_iters=2000, deadline=deadline - 1.0)
        if v_final > best_val:
            best_val = v_final
            best_h = h_final

    if best_h is None:
        best_h = np.ones(N)

    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END

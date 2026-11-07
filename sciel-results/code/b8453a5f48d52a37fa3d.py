import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill

def _value(h):
    """Compute the objective value for given step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf <= 0 or norm_1 <= 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _neg_value(h):
    """Negative of the objective for minimization."""
    v = _value(h)
    if v < 0:
        return 1e6
    return -v

def _neg_value_grad(h):
    """Gradient of negative objective (approximated numerically if needed)."""
    h = np.clip(np.asarray(h, dtype=np.float64), 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf <= 0 or norm_1 <= 0:
        return np.zeros(N)
    val = l2 / (norm_1 * norm_inf)
    
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        v_plus = _value(h_plus)
        h_minus = h.copy()
        h_minus[i] = max(0.0, h[i] - eps)
        v_minus = _value(h_minus)
        if v_plus < 0 or v_minus < 0:
            grad[i] = 0.0
        else:
            grad[i] = (v_plus - v_minus) / (2 * eps)
    return -grad  # gradient of negative objective

def _normalize(h):
    s = h.sum()
    if s <= 0:
        return np.ones(N) / N
    return h / s

def _construct_hump(power=1.0, shift=0.0, width=1.0, flat_top=0.0):
    """Concave hump centered at shift in [0,1], width relative to N, power controls shape."""
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

def _two_peak_init(peak1=0.3, peak2=0.7, power=2.0):
    """Two-peak initialization."""
    x = np.linspace(0, 1, N)
    f = np.zeros(N)
    for peak in [peak1, peak2]:
        t = np.abs(x - peak) / 0.5
        t = np.clip(t, 0, 1)
        f += (1.0 - t**power)
    f = np.maximum(f, 0.0)
    if f.sum() <= 0:
        return np.ones(N) / N
    return f / f.sum()

def _ramp_init():
    """Ramp-shaped initialization."""
    x = np.linspace(0, 1, N)
    f = x * (1 - x)  # parabolic ramp
    f = np.maximum(f, 0.0)
    if f.sum() <= 0:
        return np.ones(N) / N
    return f / f.sum()

def _alternating_init():
    """Alternating high-low initialization."""
    f = np.zeros(N)
    for i in range(N):
        if i % 2 == 0:
            f[i] = 1.0
    if f.sum() <= 0:
        return np.ones(N) / N
    return f / f.sum()

def _coordinate_polish(h0, max_iters=5000, tol=1e-9):
    """Coordinate ascent polish."""
    h = np.array(h0, dtype=float)
    h = _normalize(h)
    cur_val = _value(h)
    if cur_val < 0:
        return h, cur_val
    step = 0.05
    for it in range(max_iters):
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
            if step < 1e-4:
                break
        if it % 20 == 0:
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

def _lbfgsb_optimize(h0, max_iter=2000):
    """L-BFGS-B optimization in direct space."""
    h = np.array(h0, dtype=float)
    h = _normalize(h)
    
    bounds = [(0, None)] * N
    
    res = minimize(
        _neg_value,
        h,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    h_opt = np.clip(res.x, 0, None)
    h_opt = _normalize(h_opt)
    v_opt = _value(h_opt)
    
    if v_opt < 0:
        return h0, _value(h0)
    
    return h_opt, v_opt

def construct():
    t0 = time.time()
    
    best_h = None
    best_val = -1.0
    
    # Seed
    seed_h = np.ones(N)
    seed_val = _value(seed_h)
    best_val = seed_val
    best_h = seed_h.copy()
    
    # Diverse initializations
    inits = []
    
    # Humps with various parameters
    for power in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        inits.append(_construct_hump(power, 0.5, 1.0, 0.0))
    for flat in [0.1, 0.2, 0.3]:
        for power in [2.0, 3.0]:
            inits.append(_construct_hump(power, 0.5, 1.0, flat))
    for power in [2.0, 3.0, 4.0]:
        inits.append(_construct_hump(power, 0.5, 0.9, 0.0))
    
    # Two peaks
    for p1 in [0.25, 0.3, 0.35]:
        for p2 in [0.65, 0.7, 0.75]:
            inits.append(_two_peak_init(p1, p2, 2.0))
    
    # Ramp
    inits.append(_ramp_init())
    
    # Alternating
    inits.append(_alternating_init())
    
    # Random concave-ish shapes
    rng = np.random.default_rng(42)
    for r in range(10):
        x = np.linspace(0, 1, N)
        center = rng.uniform(0.4, 0.6)
        width = rng.uniform(0.7, 1.0)
        base = (1.0 - np.clip(np.abs(x - center) / (0.5 * width), 0, 1)) ** rng.uniform(1.5, 4.0)
        noise = rng.random(N) * 0.1
        h_init = base + noise
        h_init = np.maximum(h_init, 0.0)
        if h_init.sum() <= 0:
            h_init = np.ones(N)
        inits.append(h_init / h_init.sum())
    
    # Optimize each initialization
    for idx, h_init in enumerate(inits):
        if time.time() > DEADLINE - 10.0:
            break
        
        h_opt, v_opt = _lbfgsb_optimize(h_init, max_iter=1000)
        if v_opt > best_val:
            best_val = v_opt
            best_h = h_opt.copy()
    
    # Final polish on the best
    if best_h is not None and time.time() < DEADLINE - 5.0:
        h_final, v_final = _coordinate_polish(best_h, max_iters=8000)
        if v_final > best_val:
            best_val = v_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.ones(N)
    
    return [float(x) for x in best_h]

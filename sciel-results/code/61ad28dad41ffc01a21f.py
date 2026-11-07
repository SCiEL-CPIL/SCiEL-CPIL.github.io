"""Second autocorrelation inequality: optimize 50 nonnegative step heights.

Strategy: Direct multi-start optimization with smooth surrogate for the nonsmooth norm_inf.
The value function is scale-invariant (value = l2/(norm_1*norm_inf)), so only the shape of h matters.
We use a smooth log-sum-exp approximation for max(|g|) during optimization to give L-BFGS-B
well-behaved gradients, then compute the true value at the end.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100 spacing
TIME_LIMIT = 150.0
DEADLINE = time.time() + (TIME_LIMIT - 10.0)  # 10s safety margin
# Temperature parameter for smooth max approximation
_T = 1.0 / 20.0  # 1/T = 20


def compute_value(h):
    """Compute the true objective value (non-smooth)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    y = np.concatenate(([0.0], g, [0.0]))
    l2 = (W / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0.0 or norm_inf <= 0.0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def compute_value_smooth(h):
    """Compute a smooth surrogate for the objective using log-sum-exp for norm_inf."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    y = np.concatenate(([0.0], g, [0.0]))
    l2 = (W / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    
    # Smooth approximation of max(|g|) using log-sum-exp
    abs_g = np.abs(g)
    max_g = np.max(abs_g)
    if max_g <= 0:
        return 0.0
    # log-sum-exp: max + log(sum(exp((x - max)/T)))
    shifted = (abs_g - max_g) / _T
    # Clip to avoid overflow/underflow
    shifted = np.clip(shifted, -50, 50)
    log_sum_exp = max_g + _T * np.log(np.sum(np.exp(shifted)))
    norm_inf_smooth = log_sum_exp
    
    if norm_1 <= 0.0 or norm_inf_smooth <= 0.0:
        return 0.0
    return l2 / (norm_1 * norm_inf_smooth)


def neg_value_smooth(x):
    """Negative smooth objective for minimization."""
    s = np.sum(x)
    if s <= 0.0:
        return 1e10
    h = x / s
    return -compute_value_smooth(h)


def optimize_from(x0):
    """Optimize from x0 using smooth surrogate, return true value and h."""
    try:
        res = minimize(neg_value_smooth, x0, method="L-BFGS-B",
                       options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-10,
                                "maxls": 50})
        h = res.x / np.sum(res.x)
        h = np.clip(h, 0.0, None)
        if np.sum(h) <= 0:
            h = x0 / max(np.sum(x0), 1e-12)
        return compute_value(h), h
    except Exception:
        h = np.clip(x0, 0.0, None)
        if np.sum(h) <= 0:
            h = np.ones(N)
        return compute_value(h), h


def generate_initializations(n):
    """Generate n diverse initializations."""
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Plateau family
    for a in range(0, N // 2 + 1, 3):
        for b in np.linspace(0.0, 1.5, 8):
            h = np.empty(N)
            h[:a] = 1.0
            h[a:N - a] = b
            h[N - a:] = 1.0
            inits.append(h)
    
    # 3. Gaussian bumps
    for _ in range(30):
        t = np.linspace(0, 1, N)
        h = np.zeros(N)
        n_bumps = rng.integers(1, 5)
        for _ in range(n_bumps):
            c = rng.uniform(0.1, 0.9)
            sig = rng.uniform(0.05, 0.35)
            amp = rng.uniform(0.5, 1.5)
            h += amp * np.exp(-((t - c) ** 2) / (2 * sig ** 2))
        h += 0.05 * rng.random(N)
        inits.append(h)
    
    # 4. Cosine/sine shapes
    for _ in range(15):
        t = np.linspace(0, 1, N)
        k = rng.integers(1, 5)
        phase = rng.uniform(0, 2 * np.pi)
        h = 1.0 + rng.uniform(0.3, 1.0) * np.cos(2 * np.pi * k * t + phase)
        h = np.abs(h) + 0.1
        inits.append(h)
    
    # 5. Random smooth (interpolated)
    for _ in range(20):
        n_ctrl = rng.integers(3, 8)
        ctrl_x = np.sort(rng.uniform(0, 1, n_ctrl))
        ctrl_y = rng.uniform(0.2, 1.5, n_ctrl)
        h = np.interp(np.linspace(0, 1, N), ctrl_x, ctrl_y)
        inits.append(h)
    
    # Take first n
    return inits[:n]


def search_multi_start(n_starts=25):
    """Multi-start optimization with diverse initializations."""
    inits = generate_initializations(n_starts)
    best_v, best_h = 0.0, np.ones(N)
    
    for i, h0 in enumerate(inits):
        if time.time() > DEADLINE:
            break
        v, h = optimize_from(h0)
        if v > best_v:
            best_v, best_h = v, h.copy()
        # Progressively refine temperature for later starts
        if i % 10 == 9:
            global _T
            _T = max(_T * 0.5, 1.0 / 100.0)
    
    return best_v, best_h


def polish(h):
    """Final high-precision polish with decreasing temperature."""
    global _T
    best_v, best_h = compute_value(h), h.copy()
    
    temps = [1.0 / 50.0, 1.0 / 100.0, 1.0 / 200.0]
    for t in temps:
        if time.time() > DEADLINE:
            break
        _T = t
        v, h_new = optimize_from(best_h)
        if v > best_v:
            best_v, best_h = v, h_new
    
    # Restore default temperature
    _T = 1.0 / 20.0
    return best_v, best_h


def construct():
    global DEADLINE
    DEADLINE = time.time() + (TIME_LIMIT - 10.0)

    # 1) Multi-start with diverse initializations
    v1, h1 = search_multi_start(25)
    best_v, best_h = v1, h1.copy()

    # 2) Polish the best solution with decreasing temperature
    if time.time() < DEADLINE:
        v, h = polish(best_h)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # 3) Additional random restarts if time remains
    if time.time() < DEADLINE:
        rng = np.random.default_rng(123)
        t = np.linspace(0, 1, N)
        for _ in range(15):
            if time.time() > DEADLINE:
                break
            h0 = np.zeros(N)
            n_bumps = rng.integers(1, 4)
            for _ in range(n_bumps):
                c = rng.uniform(0.15, 0.85)
                sig = rng.uniform(0.04, 0.3)
                h0 += np.exp(-((t - c) ** 2) / (2 * sig ** 2))
            h0 += 0.03 * rng.random(N)
            v, h = optimize_from(h0)
            if v > best_v:
                best_v, best_h = v, h.copy()

    # 4) Final polish
    if time.time() < DEADLINE:
        v, h = polish(best_h)
        if v > best_v:
            best_v, best_h = v, h.copy()

    h = np.clip(best_h, 0.0, None)
    if np.sum(h) <= 0:
        h = np.ones(N)
    return [float(x) for x in h]

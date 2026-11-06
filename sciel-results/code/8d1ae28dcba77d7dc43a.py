"""
Third autocorrelation inequality - structural redesign using:
1. Randomized coordinate descent (steepest-descent direction) on the exact objective
2. Adaptive step size via line search
3. Multi-restart from diverse initializations
4. DCT spectral shaping for initialization
5. Deadline guard for 150s wall-clock
"""
import time
import numpy as np
from scipy.fft import dct as scipy_dct

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s wall-clock


def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))


def _grad_value(h):
    """
    Analytical gradient of value = |2*N*max(conv)/s^2| with respect to h.
    Returns (value, grad) where grad is d(value)/d(h).
    
    conv[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
    d(conv[k])/d(h[j]) = h[j-k] (if j>=k and j-k < N-k) + h[j+k] (if j+k < N)
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    
    # Find which lag(s) achieve the max
    max_lags = np.where(conv >= m - 1e-10)[0]
    
    # Gradient of conv[k] with respect to h
    grad_conv = np.zeros_like(h)
    for k in max_lags:
        if k == 0:
            grad_conv += 2.0 * h
        else:
            # d(conv[k])/d(h[j]) for j in [k, N-1]: h[j-k]
            grad_conv[k:] += h[:N-k]
            # d(conv[k])/d(h[j]) for j in [0, N-1-k]: h[j+k]
            grad_conv[:N-k] += h[k:]
    
    # value = |2*N*m/s^2|
    val_noabs = 2.0 * N * m / (s * s)
    
    # d(val_noabs)/d(h[j]) = (2*N/s^2)*grad_conv[j] + m*2*N*(-2)/s^3
    #                       = (2*N/s^2)*grad_conv[j] - (4*N*m/s^3)
    grad_noabs = (2.0 * N / (s * s)) * grad_conv - (4.0 * N * m / (s ** 3))
    
    val = abs(val_noabs)
    if val_noabs >= 0:
        grad = grad_noabs
    else:
        grad = -grad_noabs
    
    return val, grad


def _line_search(h, direction, step=0.01, max_iter=50):
    """
    Backtracking line search to find a good step size.
    direction: gradient direction (for minimization, we go opposite)
    """
    h_curr = h.copy()
    best_h = h_curr.copy()
    best_v = _value(h_curr)
    
    for _ in range(max_iter):
        h_try = h_curr - step * direction
        v_try = _value(h_try)
        if v_try < best_v:
            best_v = v_try
            best_h = h_try.copy()
            step *= 1.5  # increase step
        else:
            step *= 0.5  # decrease step
            if step < 1e-10:
                break
    
    return best_h, best_v


def _random_coordinate_descent(h0, max_iter=5000, deadline=None):
    """
    Randomized coordinate descent: at each step, pick a random subset of coordinates
    and do a steepest-descent update in that subspace.
    """
    h = h0.copy()
    best_h = h.copy()
    best_v = _value(h)
    
    rng = np.random.default_rng(123)
    step = 0.1
    
    for it in range(max_iter):
        if deadline is not None and (time.time() - _t0) > deadline:
            break
        
        v, grad = _grad_value(h)
        if not np.all(np.isfinite(grad)):
            break
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        grad_norm = grad / gnorm
        
        # Line search
        h_new, v_new = _line_search(h, grad_norm, step=0.01)
        
        if v_new < best_v:
            best_v = v_new
            best_h = h_new.copy()
            h = h_new
            step = min(step * 1.1, 1.0)  # gradually increase step
        else:
            step *= 0.7  # reduce step
        
        # Periodic random perturbation to escape local minima
        if it % 100 == 99 and v > best_v * 1.01:
            h += rng.normal(0, 0.05, size=N)
    
    return best_h, best_v


def _dct_init(seed, n_modes=20):
    """Generate initial h from random DCT coefficients."""
    rng = np.random.default_rng(seed)
    coeffs = np.zeros(N)
    active = rng.choice(N, size=n_modes, replace=False)
    coeffs[active] = rng.normal(0, 1.0, size=n_modes)
    # Add a DC component to ensure sum != 0
    coeffs[0] += 5.0
    h = scipy_dct(coeffs, type=2, norm='ortho')
    return h


def _golay_init(seed):
    """Generate initial h from Golay-like complementary pair."""
    rng = np.random.default_rng(seed)
    i = np.arange(N, dtype=float)
    rate1 = rng.uniform(-20, 20)
    rate2 = rng.uniform(-20, 20)
    phase1 = rng.uniform(-np.pi, np.pi)
    phase2 = rng.uniform(-np.pi, np.pi)
    amp1 = rng.uniform(0.5, 2.0)
    amp2 = rng.uniform(0.5, 2.0)
    base = rng.uniform(-1.0, 1.0)
    
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h


def _smooth_max_value_and_grad(h, beta=200.0):
    """
    Smooth approximation of max using log-sum-exp for use with L-BFGS-B.
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    log_sum_exp = m + np.log(np.sum(exp_scaled)) / beta
    
    weights = exp_scaled / np.sum(exp_scaled)
    
    grad_h = np.zeros_like(h)
    for k in range(len(conv)):
        if weights[k] < 1e-12:
            continue
        w = weights[k]
        if k == 0:
            grad_h += w * 2.0 * h
        else:
            grad_h[k:] += w * h[:N-k]
            grad_h[:N-k] += w * h[k:]
    
    val_noabs = 2.0 * N * log_sum_exp / (s * s)
    if val_noabs >= 0:
        grad = (2.0 * N / (s * s)) * grad_h - (4.0 * N / (s ** 3)) * log_sum_exp
    else:
        grad = -( (2.0 * N / (s * s)) * grad_h - (4.0 * N / (s ** 3)) * log_sum_exp )
    
    return abs(val_noabs), grad


def _lbfgsb_refine(h0, max_iter=300, beta=200.0, deadline=None):
    """L-BFGS-B refinement with smooth max surrogate."""
    from scipy.optimize import minimize
    
    def obj_and_grad(h):
        val, grad = _smooth_max_value_and_grad(h, beta)
        if not np.all(np.isfinite(grad)):
            grad = np.zeros_like(h)
        return val, grad
    
    res = minimize(obj_and_grad, h0, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12,
                            'maxls': 50})
    if res.x is not None and np.all(np.isfinite(res.x)):
        return res.x, res.fun
    return h0, _smooth_max_value_and_grad(h0, beta)[0]


def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: Multiple random initializations with coordinate descent
    n_restarts = 8
    for r in range(n_restarts):
        if not deadline_ok():
            break
        
        # Alternate between DCT and Golay initializations
        if r % 2 == 0:
            h0 = _dct_init(seed=100 + r, n_modes=15 + r * 3)
        else:
            h0 = _golay_init(seed=200 + r)
        
        # Ensure sum is not zero
        if abs(np.sum(h0)) < 1e-6:
            h0 += 1.0
        
        h_cd, v_cd = _random_coordinate_descent(h0, max_iter=3000, deadline=_DEADLINE)
        update_best(h_cd)
        
        if not deadline_ok():
            break
    
    # Phase 2: L-BFGS-B refinement on best candidate
    if deadline_ok():
        try:
            h_lbfgs, v_lbfgs = _lbfgsb_refine(best_h, max_iter=400, beta=200.0, deadline=_DEADLINE)
            update_best(h_lbfgs)
        except Exception:
            pass
    
    # Phase 3: More coordinate descent from refined solution
    if deadline_ok():
        try:
            h_cd2, v_cd2 = _random_coordinate_descent(best_h, max_iter=5000, deadline=_DEADLINE)
            update_best(h_cd2)
        except Exception:
            pass
    
    # Phase 4: Final L-BFGS-B polish with higher beta
    if deadline_ok():
        try:
            h_lbfgs2, v_lbfgs2 = _lbfgsb_refine(best_h, max_iter=300, beta=300.0, deadline=_DEADLINE)
            update_best(h_lbfgs2)
        except Exception:
            pass
    
    # Phase 5: One more round of coordinate descent
    if deadline_ok():
        try:
            h_cd3, v_cd3 = _random_coordinate_descent(best_h, max_iter=3000, deadline=_DEADLINE)
            update_best(h_cd3)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h

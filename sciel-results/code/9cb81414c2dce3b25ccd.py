import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _ratio(h, beta=100.0):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf) using soft-max for peak."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 1e-15:
        return 0.0
    # Soft-max approximation of max using log-sum-exp
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted))
    soft_max = max_val + log_sum_exp / beta
    if soft_max <= 1e-15:
        return 0.0
    return l2_sq / soft_max

def _ratio_neg(h, beta=100.0):
    return -_ratio(h, beta)

def _gaussian_profile(sigma, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return h

def _lorentzian_profile(gamma, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return h

def _sinc_like_profile(width, center=0.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = (x - center) / width
    h = np.sinc(t) ** 2
    return np.maximum(h, 0)

def _two_gaussian(sigma1, center1, sigma2, center2):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h1 = np.exp(-((x - center1) ** 2) / (2 * sigma1 ** 2))
    h2 = np.exp(-((x - center2) ** 2) / (2 * sigma2 ** 2))
    return h1 + h2

def _trapezoid_profile(w1, w2, center=0.0):
    """Trapezoid profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = np.abs(x - center)
    h = np.zeros(N)
    mask1 = t <= w1
    mask2 = (t > w1) & (t <= w2)
    h[mask1] = 1.0
    h[mask2] = 1.0 - (t[mask2] - w1) / (w2 - w1)
    return np.maximum(h, 0)

def _power_profile(k, side='right'):
    """Power law profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = np.zeros(N)
    if side == 'right':
        mask = x > 0
        h[mask] = x[mask] ** k
    elif side == 'left':
        mask = x < 0
        h[mask] = (-x[mask]) ** k
    else:
        h = np.abs(x) ** k
    return h

def construct():
    start_time = time.time()
    best_heights = np.ones(N)
    best_score = _ratio(best_heights)
    
    # Generate diverse initializations as h profiles
    initializations = []
    
    # Gaussian profiles
    for sigma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        for center in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]:
            h = _gaussian_profile(sigma, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Lorentzian profiles
    for gamma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        for center in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
            h = _lorentzian_profile(gamma, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Sinc-like profiles
    for width in [0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
        for center in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            h = _sinc_like_profile(width, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Two-Gaussian profiles
    for params in [
        (0.02, -0.15, 0.02, 0.15),
        (0.03, -0.1, 0.03, 0.1),
        (0.03, -0.2, 0.03, 0.2),
        (0.05, -0.15, 0.05, 0.15),
        (0.05, -0.1, 0.05, 0.1),
        (0.05, -0.05, 0.05, 0.05),
        (0.08, -0.1, 0.08, 0.1),
        (0.02, 0.0, 0.1, 0.0),
        (0.03, -0.2, 0.05, 0.1),
        (0.02, -0.1, 0.03, 0.15),
    ]:
        h = _two_gaussian(*params)
        if np.sum(h) > 0:
            initializations.append(h.copy())
    
    # Trapezoid profiles
    for w1, w2 in [(0.01, 0.05), (0.02, 0.1), (0.05, 0.15), (0.01, 0.03), (0.03, 0.08)]:
        for center in [-0.1, -0.05, 0.0, 0.05, 0.1]:
            h = _trapezoid_profile(w1, w2, center)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Power law profiles
    for k in [2, 3, 4, 5, 6, 8]:
        for side in ['right', 'left', 'both']:
            h = _power_profile(k, side)
            if np.sum(h) > 0:
                initializations.append(h.copy())
    
    # Uniform
    initializations.append(np.ones(N).copy())
    
    # Random profiles
    rng = np.random.RandomState(42)
    for _ in range(15):
        h = rng.rand(N) ** rng.choice([1, 2, 3, 4])
        if np.sum(h) > 0:
            initializations.append(h.copy())
    
    # Sort initializations by score, keep top ones for optimization
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0.0)
        l1 = np.sum(h_init)
        if l1 < 1e-15:
            continue
        h_init = h_init / l1
        s = _ratio(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best from initializations
    if scored_inits and scored_inits[0][0] > best_score:
        best_score = scored_inits[0][0]
        best_heights = scored_inits[0][1].copy()
    
    # Optimize from top initializations using L-BFGS-B
    top_k = min(20, len(scored_inits))
    for i in range(top_k):
        if _time_left(start_time) < 8:
            break
        
        s_init, h_init = scored_inits[i]
        
        # Direct optimization with L-BFGS-B
        try:
            result = minimize(
                _ratio_neg, h_init, args=(100.0,),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
            )
            h_opt = np.maximum(result.x, 0.0)
            l1_opt = np.sum(h_opt)
            if l1_opt < 1e-15:
                continue
            h_opt = h_opt / l1_opt
            score_opt = _ratio(h_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_heights = h_opt.copy()
        except Exception:
            continue
        
        # Local perturbation of best found so far
        if _time_left(start_time) > 15 and best_score > 0:
            for _ in range(2):
                if _time_left(start_time) < 8:
                    break
                h_pert = best_heights + 0.02 * rng.randn(N)
                h_pert = np.maximum(h_pert, 0.0)
                l1_pert = np.sum(h_pert)
                if l1_pert < 1e-15:
                    continue
                h_pert = h_pert / l1_pert
                try:
                    result = minimize(
                        _ratio_neg, h_pert, args=(100.0,),
                        method='L-BFGS-B',
                        bounds=[(0, None)] * N,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
                    )
                    h_opt = np.maximum(result.x, 0.0)
                    l1_opt = np.sum(h_opt)
                    if l1_opt < 1e-15:
                        continue
                    h_opt = h_opt / l1_opt
                    score_opt = _ratio(h_opt)
                    
                    if score_opt > best_score:
                        best_score = score_opt
                        best_heights = h_opt.copy()
                except Exception:
                    continue
    
    # Final polish: high-precision optimization from best found
    if _time_left(start_time) > 12:
        try:
            result = minimize(
                _ratio_neg, best_heights, args=(100.0,),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            h_polish = np.maximum(result.x, 0.0)
            l1_polish = np.sum(h_polish)
            if l1_polish < 1e-15:
                h_polish = best_heights
            else:
                h_polish = h_polish / l1_polish
                score_polish = _ratio(h_polish)
                if score_polish > best_score:
                    best_score = score_polish
                    best_heights = h_polish.copy()
        except Exception:
            pass
    
    # Final safety: ensure we return valid non-negative heights
    if best_score <= 0:
        best_heights = np.ones(N)
    
    return best_heights.tolist()

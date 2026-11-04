import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 150s kill

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
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    # Soft-max approximation of max using log-sum-exp
    shifted = beta * (conv - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * max_val) / beta
    if soft_max <= 0:
        return 0.0
    return l2_sq / soft_max

def _ratio_neg(h, beta=100.0):
    return -_ratio(h, beta)

def _ratio_and_grad(h, beta=100.0):
    """Compute ratio and gradient via finite differences (central)."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0, np.zeros(N)
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0, np.zeros(N)
    
    shifted = beta * (conv - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * max_val) / beta
    if soft_max <= 0:
        return 0.0, np.zeros(N)
    
    ratio = l2_sq / soft_max
    
    # Gradient via central finite differences
    eps = 1e-6
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] = max(0.0, h_minus[i] - eps)
        grad[i] = (_ratio(h_plus, beta) - _ratio(h_minus, beta)) / (2 * eps)
    return ratio, grad

def _local_optimize(h0, start_time, max_iter=500, tol=1e-10, beta=100.0):
    """Direct optimization of h with non-negativity constraint."""
    h = np.maximum(h0, 0.0).copy()
    
    def objective(h_flat):
        return _ratio_neg(h_flat, beta)
    
    result = minimize(objective, h, method='L-BFGS-B',
                      bounds=[(0, None)] * N,
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 30})
    
    h_opt = np.maximum(result.x, 0.0)
    return h_opt, -result.fun

def _gaussian_profile(center, width, height):
    """Create a Gaussian-like profile on the 50 bins."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    return h

def _lorentzian_profile(center, gamma, height):
    """Create a Lorentzian-like profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = height * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)
    return h

def _power_profile(k, side='right'):
    """Power law profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
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

def _two_gaussian(c1, w1, c2, w2, h1, h2):
    """Sum of two Gaussians."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h1_arr = h1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
    h2_arr = h2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2))
    return h1_arr + h2_arr

def _sinc_like_profile(center, width, height):
    """Sinc-like profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    t = (x - center) / width
    h = height * np.sinc(t) ** 2
    h = np.maximum(h, 0.0)
    return h

def _triangle_profile(center, width, height):
    """Triangle profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    h = height * np.maximum(0, 1 - np.abs(x - center) / width)
    return h

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Generate diverse initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N))
    
    # Single Gaussians at various positions and widths
    for center in [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]:
        for width in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
            initializations.append(_gaussian_profile(center, width, 1.0))
    
    # Lorentzians
    for center in [-0.15, -0.1, 0.0, 0.1, 0.15]:
        for gamma in [0.03, 0.05, 0.08, 0.1, 0.15]:
            initializations.append(_lorentzian_profile(center, gamma, 1.0))
    
    # Power laws
    for k in [2, 4, 6, 8, 10]:
        initializations.append(_power_profile(k, 'right'))
        initializations.append(_power_profile(k, 'left'))
        initializations.append(_power_profile(k, 'both'))
    
    # Two Gaussians
    for c1, c2 in [(-0.15, 0.15), (-0.1, 0.1), (-0.2, 0.2), (-0.12, 0.12), (-0.08, 0.08)]:
        for w in [0.04, 0.06, 0.08, 0.1, 0.12]:
            initializations.append(_two_gaussian(c1, w, c2, w, 1.0, 1.0))
    
    # Sinc-like
    for center in [-0.1, 0.0, 0.1]:
        for width in [0.05, 0.1, 0.15]:
            initializations.append(_sinc_like_profile(center, width, 1.0))
    
    # Triangle
    for center in [-0.1, 0.0, 0.1]:
        for width in [0.05, 0.1, 0.15, 0.2]:
            initializations.append(_triangle_profile(center, width, 1.0))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(20):
        h_rand = rng.random(N)
        initializations.append(h_rand)
    
    # Sparse random
    for _ in range(10):
        h_sparse = np.zeros(N)
        idx = rng.choice(N, size=rng.randint(5, 15), replace=False)
        h_sparse[idx] = rng.random(len(idx))
        initializations.append(h_sparse)
    
    # Sort initializations by score, keep top ones for optimization
    scored_inits = []
    for h_init in initializations:
        h_init = np.maximum(h_init, 0.0)
        s = _ratio(h_init)
        scored_inits.append((s, h_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best from initializations
    if scored_inits[0][0] > best_score:
        best_score = scored_inits[0][0]
        best_h = scored_inits[0][1].copy()
    
    # Optimize from top initializations
    top_k = min(20, len(scored_inits))
    for i in range(top_k):
        if _time_left(start_time) < 5.0:
            break
        
        s_init, h_init = scored_inits[i]
        
        # Direct optimization with high beta
        h_opt, s_opt = _local_optimize(h_init, start_time, max_iter=300, tol=1e-10, beta=100.0)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Final polish: high-precision optimization from best found with multiple betas
    if _time_left(start_time) > 10.0:
        for beta in [100.0, 200.0, 500.0]:
            if _time_left(start_time) < 5.0:
                break
            h_polish, s_polish = _local_optimize(best_h, start_time, max_iter=500, tol=1e-12, beta=beta)
            if s_polish > best_score:
                best_score = s_polish
                best_h = h_polish.copy()
    
    # One more polish with very high beta for near-exact max
    if _time_left(start_time) > 5.0:
        def objective_final(h_flat):
            return -_ratio(h_flat, beta=1000.0)
        
        result = minimize(objective_final, best_h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'gtol': 1e-12, 'maxls': 30})
        h_final = np.maximum(result.x, 0.0)
        s_final = _ratio(h_final, beta=1000.0)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    # Final exact evaluation
    h_final = np.maximum(best_h, 0.0)
    l1 = np.sum(h_final)
    if l1 > 0:
        h_norm = h_final / l1
        l2_sq = np.sum(h_norm ** 2)
        conv = _autoconv(h_norm)
        max_val = np.max(conv)
        if max_val > 0:
            exact_score = l2_sq / max_val
            if exact_score > best_score:
                best_score = exact_score
    
    return best_h.tolist()

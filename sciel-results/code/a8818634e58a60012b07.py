import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * Sup(AutoConv))"""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _smooth_max_conv(h, beta=100.0):
    """Smooth approximation of max(conv(h,h)) using log-sum-exp.
    Higher beta = closer to true max."""
    h_conv = np.convolve(h, h, mode='full')
    # Use log-sum-exp for numerical stability
    max_val = np.max(h_conv)
    if max_val <= 0:
        return 0.0
    shifted = h_conv / max_val
    log_sum_exp = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
    return max_val * np.exp(log_sum_exp)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Quick grid search for good initializations (reduced to save time)
    # 1. Uniform (already done)
    
    # 2. Gaussian - reduced grid
    for c in np.linspace(10, N-10, 5):
        for w in np.linspace(2, N/3, 5):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
    
    # 3. Triangular - reduced grid
    for c in np.linspace(10, N-10, 5):
        for w in np.linspace(2, N/2, 5):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 4. Random sparse initialization (reduced)
    for _ in range(30):
        if time.time() > deadline - 60: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using smooth surrogate for max(conv)
    # Objective: Maximize L2^2 / SmoothMaxConv(h) subject to L1=1, h>=0
    # This is equivalent to maximizing _score_raw(h) / DELTA with L1=1
    
    def neg_obj_smooth(h, beta=100.0):
        """Smooth objective using log-sum-exp for max(conv)"""
        h = np.maximum(h, 0)  # Ensure non-negative
        l1 = np.sum(h)
        if l1 == 0: return 0
        h_norm = h / l1  # Normalize to L1=1
        l2_sq = np.sum(h_norm**2)
        smooth_max = _smooth_max_conv(h_norm, beta)
        if smooth_max == 0: return 0
        return -(l2_sq / (DELTA * smooth_max))
    
    def neg_obj_exact(h):
        """Exact objective for final evaluation"""
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 == 0: return 0
        h_norm = h / l1
        return -_score_raw(h_norm)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    bounds = [(0.0, 1.0)] * N
    
    # Phase 1: Optimize with smooth surrogate from best_h
    h_start = best_h / np.sum(best_h)
    try:
        res = minimize(neg_obj_smooth, h_start, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
        if res.success:
            h_candidate = res.x
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Phase 2: Multiple restarts with smooth surrogate
    inits = [best_h / np.sum(best_h)]
    
    # Add diverse initializations
    inits.append(np.ones(N) / N)
    for c in [N//4, N//2, 3*N//4]:
        for w in [3, 5, 10]:
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
    
    for _ in range(10):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
    
    for h_init in inits:
        if time.time() > deadline - 30: break
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(neg_obj_smooth, h_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with exact objective (tighter tolerance)
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_exact, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure final result is valid and normalized
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)

# EVOLVE-BLOCK-END

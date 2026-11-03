import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.all(h == 0):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = None
    best_score = 0.0
    rng = np.random.RandomState(42)
    
    def optimize_support(support_indices):
        """Optimize heights on a given support using Nelder-Mead."""
        k = len(support_indices)
        if k == 0:
            return np.zeros(N), 0.0
        
        h_init = np.zeros(N)
        h_init[support_indices] = 1.0
        
        def neg_ratio(x):
            h_full = np.zeros(N)
            h_full[support_indices] = x
            return -_compute_ratio(h_full)
        
        try:
            res = minimize(neg_ratio, h_init[support_indices], method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7})
            h_opt = np.zeros(N)
            h_opt[support_indices] = res.x
            score = -res.fun
            return h_opt, score
        except Exception:
            return h_init, _compute_ratio(h_init)

    # Try various support sizes and random configurations
    # Larger supports generally allow for higher ratios if the autoconvolution peak is controlled
    support_sizes = [10, 15, 20, 25, 30, 40]
    
    for k in support_sizes:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random supports for each size
        num_trials = 5 if k < 20 else 3
        
        for _ in range(num_trials):
            if time.time() > deadline - 10:
                break
            support = rng.choice(N, size=k, replace=False)
            h_opt, score = optimize_support(support)
            if score > best_score:
                best_score = score
                best_h = h_opt

    # If we have time, try a full vector optimization from the best sparse solution
    if best_h is not None and time.time() < deadline - 5:
        def neg_ratio_full(x):
            return -_compute_ratio(x)
        
        try:
            # Use Nelder-Mead on the full vector, starting from best_h
            # This allows non-zero values at previously zero indices
            res = minimize(neg_ratio_full, best_h, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            h_polished = res.x
            score_polished = -res.fun
            if score_polished > best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass

    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)

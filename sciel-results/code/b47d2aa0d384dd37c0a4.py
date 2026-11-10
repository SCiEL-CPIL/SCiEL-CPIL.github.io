# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp


def _shannon_entropy_bits(probs):
    """Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))

def _precompute_bins(n=10):
    bins = {}
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # H_0: x
    bins['x'] = X.flatten().astype(int)
    
    # H_{-1}: x - y
    vals = (X - Y).flatten()
    min_val = vals.min()
    bins['x_minus_y'] = (vals - min_val).astype(int)
    
    # H_1: x + y
    vals = (X + Y).flatten()
    min_val = vals.min()
    bins['x_plus_y'] = (vals - min_val).astype(int)
    
    # H_2: x + 2y
    vals = (X + 2*Y).flatten()
    min_val = vals.min()
    bins['x_plus_2y'] = (vals - min_val).astype(int)
    
    return bins

BINS = _precompute_bins(10)

def _compute_ratio_fast(P):
    P_flat = P.flatten()
    
    # H_{-1}
    probs_minus1 = np.bincount(BINS['x_minus_y'], weights=P_flat, minlength=19)
    H_minus1 = _shannon_entropy_bits(probs_minus1)
    
    # H_0
    probs_0 = np.bincount(BINS['x'], weights=P_flat, minlength=10)
    H_0 = _shannon_entropy_bits(probs_0)
    
    # H_1
    probs_1 = np.bincount(BINS['x_plus_y'], weights=P_flat, minlength=19)
    H_1 = _shannon_entropy_bits(probs_1)
    
    # H_2
    probs_2 = np.bincount(BINS['x_plus_2y'], weights=P_flat, minlength=28)
    H_2 = _shannon_entropy_bits(probs_2)
    
    M = max(H_0, H_1, H_2)
    if M < 1e-9:
        return 0.0
    return H_minus1 / M

def _optimize_fixed_support(S, n=10, max_iter=200):
    k = len(S)
    if k == 0:
        return None, 0.0
    
    def obj(params):
        log_w = params
        log_w_shifted = log_w - np.max(log_w)
        w = np.exp(log_w_shifted)
        w = w / w.sum()
        
        P = np.zeros((n, n))
        for i, idx in enumerate(S):
            P[idx] = w[i]
            
        return -_compute_ratio_fast(P)
    
    x0 = np.zeros(k)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success:
            log_w_opt = res.x
            log_w_opt_shifted = log_w_opt - np.max(log_w_opt)
            w_opt = np.exp(log_w_opt_shifted)
            w_opt = w_opt / w_opt.sum()
            
            P_opt = np.zeros((n, n))
            for i, idx in enumerate(S):
                P_opt[idx] = w_opt[i]
            
            ratio = _compute_ratio_fast(P_opt)
            return P_opt, ratio
    except Exception:
        pass
        
    try:
        res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success:
            log_w_opt = res.x
            log_w_opt_shifted = log_w_opt - np.max(log_w_opt)
            w_opt = np.exp(log_w_opt_shifted)
            w_opt = w_opt / w_opt.sum()
            
            P_opt = np.zeros((n, n))
            for i, idx in enumerate(S):
                P_opt[idx] = w_opt[i]
            
            ratio = _compute_ratio_fast(P_opt)
            return P_opt, ratio
    except Exception:
        pass
        
    return None, 0.0

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 600 - 30
    
    P0 = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P0.copy()
    best_ratio = _compute_ratio_fast(P0)
    
    np.random.seed(42)
    all_cells = [(y, x) for y in range(n) for x in range(n)]
    
    # Sparse Support Search
    while time.time() < deadline - 60:
        # Randomly select support size
        size = np.random.choice([10, 15, 20, 25, 30])
        
        # Random support
        support_indices = np.random.choice(len(all_cells), size=size, replace=False)
        S = [all_cells[i] for i in support_indices]
        
        P_opt, ratio_opt = _optimize_fixed_support(S)
        
        if P_opt is not None and ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt.copy()
            
            # If we found a good one, maybe try to refine it with full optimization
            # But full optimization is slow. Let's just do a few more sparse trials.
            
    # Final polish with full optimization on the best found support structure
    # Extract support from best_P
    support_indices = np.where(best_P > 1e-6)[0]
    # Map flat indices to (y, x)
    S_best = []
    for idx in support_indices:
        y, x = np.unravel_index(idx, (n, n))
        S_best.append((y, x))
    
    if len(S_best) > 0:
        # Optimize weights on this specific support
        P_refined, ratio_refined = _optimize_fixed_support(S_best, max_iter=500)
        if P_refined is not None and ratio_refined > best_ratio:
            best_ratio = ratio_refined
            best_P = P_refined.copy()
            
    # One last full optimization attempt if time permits
    if time.time() < deadline - 10:
        log_P_init = np.log(best_P + 1e-15)
        
        def _objective_general(params, n=10):
            log_P = params.reshape(n, n)
            log_P_shifted = log_P - np.max(log_P)
            P = np.exp(log_P_shifted)
            s = P.sum()
            if s < 1e-15:
                return 1e6
            P = P / s
            return -_compute_ratio_fast(P)
        
        try:
            result = minimize(
                _objective_general,
                log_P_init.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100},
                args=(n,)
            )
            if result.fun < -best_ratio:
                log_P_opt = result.x.reshape(n, n)
                log_P_shifted = log_P_opt - np.max(log_P_opt)
                P_opt = np.exp(log_P_shifted)
                s_opt = P_opt.sum()
                if s_opt > 1e-15:
                    P_opt = P_opt / s_opt
                    ratio_opt = _compute_ratio_fast(P_opt)
                    if ratio_opt > best_ratio:
                        best_ratio = ratio_opt
                        best_P = P_opt.copy()
        except Exception:
            pass

    best_P = np.clip(best_P, 0, None)
    s_final = best_P.sum()
    if s_final < 1e-9:
        best_P = P0
    else:
        best_P = best_P / s_final
    
    return best_P

# EVOLVE-BLOCK-END

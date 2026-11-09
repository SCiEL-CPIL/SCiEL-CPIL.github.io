# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    P = sum_{k,l} w_k w_l * Prob[X1+X2 < 2*s_l - s_k]
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Generate all pair sums and their probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # (n, n)
    pair_probs = np.outer(w, w) # (n, n)
    
    # Flatten
    sums_flat = pair_sums.flatten() # (n^2,)
    probs_flat = pair_probs.flatten() # (n^2,)
    
    # Sort sums and compute cumulative probabilities
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # Ensure cum_probs ends at 1.0 (floating point safety)
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
             cum_probs[-1] = 1.0
    
    # Compute P = sum_{k,l} w_k w_l * CDF(2*s_l - s_k)
    thresholds = 2.0 * s_row - s_col  # (n, n)
    
    # For each threshold, find the count of pair_sums < threshold
    # searchsorted with side='left' gives the index of the first element >= threshold.
    # We want sum of probs for elements < threshold.
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get CDF values
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    # Sum over k, l with weights w_k w_l
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy:
    1. Global Grid Search for n=3 support points.
       We assume the optimal distribution has 3 points: 0, x, 1 (or similar scaling).
       Actually, scaling invariance: P[X1+X2+X3 < 2X4] is scale invariant?
       Let's check: If we scale all s by c, the inequality becomes c(X1+X2+X3) < c(2X4).
       So P is invariant to scaling of the support.
       Thus, we can fix the support to include 0 and 1, and optimize the middle point x in [0,1]
       and the weights w0, w1, w2 (summing to 1).
       
       However, the optimal support might not be [0, x, 1]. It could be [0, x, y].
       Due to scale invariance, we can fix the max support to 1.
       So we search over x1, x2 in [0, 1] (with x1 < x2) and weights.
       
       Actually, let's just do a coarse grid search over 3 support points s0, s1, s2
       and 3 weights w0, w1, w2.
       
       To make it efficient:
       Fix s0 = 0 (likely optimal to have 0).
       Fix s2 = 1 (scale invariance).
       Vary s1 in [0, 1].
       Vary w0, w1, w2.
       
       Wait, is s0=0 always optimal?
       For n=2, optimal is [0, 1] with weights ~[0.618, 0.382].
       For n=3, likely [0, x, 1].
       
       Let's do a grid search:
       s1 in linspace(0.1, 0.9, 10)
       w1 in linspace(0.1, 0.8, 10)
       w0 in linspace(0.1, 0.8, 10) such that w0+w1+w2=1 => w2 = 1-w0-w1 >= 0.
       
       This is 10 * 10 * 10 = 1000 evaluations. Each evaluation is O(n^2 log n^2) ~ O(9 log 9) ~ trivial.
       This is extremely fast.
       
       Then, take the best candidate and polish with Nelder-Mead on the full 3x2 = 6 parameters
       (log_w and log_s, or just s and w with constraints).
       Nelder-Mead is derivative-free and robust to the non-smoothness of the CDF steps.
    """
    t_start = time.time()
    deadline = t_start + 600.0
    
    best_P_global = -1.0
    best_support_global = None
    best_weights_global = None
    
    def update_best(s, w, P_val):
        nonlocal best_P_global, best_support_global, best_weights_global
        if P_val > best_P_global:
            best_P_global = P_val
            best_support_global = s.copy()
            best_weights_global = w.copy()
            
    # --- Phase 1: Grid Search for n=3 ---
    # Assume support is [0, x, 1].
    # Weights w0, w1, w2.
    
    # Grid for x (middle support)
    x_vals = np.linspace(0.05, 0.95, 20)
    
    # Grid for w1 (weight of middle support)
    # w0 + w1 + w2 = 1
    # Let's sample w1 and w0
    w1_vals = np.linspace(0.05, 0.8, 20)
    w0_vals = np.linspace(0.05, 0.8, 20)
    
    # We can prune w0_vals based on w1 to ensure w2 > 0
    # But 20*20*20 = 8000 is small.
    
    for x in x_vals:
        s_grid = np.array([0.0, x, 1.0])
        for w1 in w1_vals:
            for w0 in w0_vals:
                w2 = 1.0 - w0 - w1
                if w2 < 1e-6:
                    continue
                w_grid = np.array([w0, w1, w2])
                
                P_val = _compute_P_fast(s_grid, w_grid)
                if P_val > best_P_global:
                    update_best(s_grid, w_grid, P_val)
                    
    # Check time
    if time.time() > deadline - 120:
        # Skip polish if tight, but 8000 evals should be < 1s
        pass

    # --- Phase 2: Local Polish with Nelder-Mead ---
    # Use the best point from grid search as initial guess.
    # Parameters: s1, s2 (s0=0 fixed? No, let's optimize all 3 s and 3 w)
    # Actually, fixing s0=0 and s2=1 is a strong assumption.
    # Let's try optimizing all 6 parameters: s0, s1, s2, w0, w1, w2.
    # Constraints: s0 < s1 < s2, w_i > 0, sum(w)=1.
    # Nelder-Mead can handle unconstrained if we map parameters.
    # Map:
    # s_i = exp(log_s_i)
    # w_i = softmax(log_w_i)
    
    if best_support_global is not None:
        s_init = best_support_global
        w_init = best_weights_global
        
        # Ensure s is positive for log
        s_init_safe = np.maximum(s_init, 1e-6)
        w_init_safe = np.maximum(w_init, 1e-6)
        
        log_s_init = np.log(s_init_safe)
        log_w_init = np.log(w_init_safe)
        
        x0 = np.concatenate([log_w_init, log_s_init])
        
        def objective_polish(x):
            lw = x[:3]
            ls = x[3:]
            
            # Decode weights
            lw_max = np.max(lw)
            w = np.exp(lw - lw_max)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 0.0
            w = w / w_sum
            
            # Decode support
            s = np.exp(ls)
            s = np.maximum(s, 0.0)
            
            # Sort to ensure consistent ordering?
            # _compute_P_fast handles any order, but sorting might help stability
            sort_idx = np.argsort(s)
            s_sorted = s[sort_idx]
            w_sorted = w[sort_idx]
            
            P = _compute_P_fast(s_sorted, w_sorted)
            return -P
        
        # Nelder-Mead options
        options_nm = {
            'maxiter': 1000,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
        
        try:
            res = minimize(objective_polish, x0, method='Nelder-Mead', options=options_nm)
            if res.fun < objective_polish(x0):
                final_log_w = res.x[:3]
                final_log_s = res.x[3:]
                
                # Decode
                lw_max = np.max(final_log_w)
                w = np.exp(final_log_w - lw_max)
                w = w / np.sum(w)
                
                s = np.exp(final_log_s)
                s = np.maximum(s, 0.0)
                
                sort_idx = np.argsort(s)
                s_sorted = s[sort_idx]
                w_sorted = w[sort_idx]
                
                P_val = _compute_P_fast(s_sorted, w_sorted)
                update_best(s_sorted, w_sorted, P_val)
        except Exception:
            pass

    # --- Phase 3: Try n=4 with a few random restarts if time permits ---
    if time.time() < deadline - 60:
        np.random.seed(42)
        for _ in range(5):
            # Random init for n=4
            s_rand = np.sort(np.random.uniform(0, 1, 4))
            w_rand = np.random.uniform(0.1, 1.0, 4)
            w_rand = w_rand / np.sum(w_rand)
            
            P_val = _compute_P_fast(s_rand, w_rand)
            if P_val > best_P_global:
                update_best(s_rand, w_rand, P_val)
                
            # Quick polish for n=4
            if time.time() < deadline - 30:
                s_init_safe = np.maximum(s_rand, 1e-6)
                w_init_safe = np.maximum(w_rand, 1e-6)
                log_s_init = np.log(s_init_safe)
                log_w_init = np.log(w_init_safe)
                x0 = np.concatenate([log_w_init, log_s_init])
                
                def objective_polish_4(x):
                    lw = x[:4]
                    ls = x[4:]
                    lw_max = np.max(lw)
                    w = np.exp(lw - lw_max)
                    w_sum = np.sum(w)
                    if w_sum < 1e-9:
                        return 0.0
                    w = w / w_sum
                    s = np.exp(ls)
                    s = np.maximum(s, 0.0)
                    sort_idx = np.argsort(s)
                    s_sorted = s[sort_idx]
                    w_sorted = w[sort_idx]
                    P = _compute_P_fast(s_sorted, w_sorted)
                    return -P
                
                try:
                    res = minimize(objective_polish_4, x0, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                    if res.fun < objective_polish_4(x0):
                        final_log_w = res.x[:4]
                        final_log_s = res.x[4:]
                        lw_max = np.max(final_log_w)
                        w = np.exp(final_log_w - lw_max)
                        w = w / np.sum(w)
                        s = np.exp(final_log_s)
                        s = np.maximum(s, 0.0)
                        sort_idx = np.argsort(s)
                        s_sorted = s[sort_idx]
                        w_sorted = w[sort_idx]
                        P_val = _compute_P_fast(s_sorted, w_sorted)
                        update_best(s_sorted, w_sorted, P_val)
                except Exception:
                    pass

    # Final Validation
    if best_support_global is None or len(best_support_global) == 0:
        # Fallback
        return np.array([0.0, 1.0]), np.array([0.618, 0.382])
        
    best_support_global = np.maximum(best_support_global, 0.0)
    best_weights_global = np.maximum(best_weights_global, 0.0)
    
    w_sum = np.sum(best_weights_global)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.618, 0.382])
        
    best_weights_global = best_weights_global / w_sum
    
    if not (np.all(np.isfinite(best_support_global)) and np.all(np.isfinite(best_weights_global))):
        return np.array([0.0, 1.0]), np.array([0.618, 0.382])
        
    return best_support_global, best_weights_global

# EVOLVE-BLOCK-END

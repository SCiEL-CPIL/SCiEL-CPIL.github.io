# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    P = sum_{k,l} w_k w_l * Prob[X1+X2 < 2*s_l - s_k]
    
    We precompute the distribution of S = X1 + X2.
    S takes values s_i + s_j with probability w_i w_j.
    There are n^2 such values.
    For each pair (k, l), we need the CDF of S at threshold t = 2*s_l - s_k.
    Since S is a discrete variable with n^2 atoms, we can sort the atoms once,
    and then for each of the n^2 thresholds, do a binary search (or vectorized search).
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
    # s_col: (n, 1), s_row: (1, n)
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
        # Adjust last element to 1.0 if it's close
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
             cum_probs[-1] = 1.0
    
    # Now compute P = sum_{k,l} w_k w_l * CDF(2*s_l - s_k)
    # CDF(t) = sum_{s_i+s_j < t} P(s_i+s_j)
    # Using np.searchsorted with side='left' gives the index of the first element >= t.
    # So the number of elements < t is the index itself.
    # The probability mass is cum_probs[index - 1] if index > 0, else 0.
    
    # We need to do this for all n^2 pairs of (k, l).
    # s_k: (n, 1), s_l: (1, n) -> thresholds: (n, n)
    thresholds = 2.0 * s_row - s_col  # (n, n)
    
    # For each threshold, find the count of pair_sums < threshold
    # searchsorted(sorted_sums, thresholds, side='left') returns indices (n, n)
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get CDF values
    # If index is 0, CDF is 0. If index is i, CDF is cum_probs[i-1].
    # We can handle this by prepending 0 to cum_probs.
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    # Now sum over k, l with weights w_k w_l
    # P = sum_{k,l} w_k w_l cdf_values[k,l]
    # Note: s_col is (n,1) so k index is rows, s_row is (1,n) so l index is cols.
    # w_k is (n,1), w_l is (1,n)
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. 
    Uses Differential Evolution to optimize support and weights jointly.
    """
    t_start = time.time()
    deadline = t_start + 600.0
    
    # Parameters for DE
    # We will optimize n points. 
    # Let's try n=4, 5, 6. 
    # The condition X1+X2+X3 < 2X4 suggests a bimodal or multimodal distribution might work best.
    # Specifically, having a mass at 0 helps (if X1,X2,X3 are small).
    # Having a mass at a large value helps (if X4 is large).
    
    # Strategy: 
    # 1. Try n=3, 4, 5, 6 with DE.
    # 2. For each n, define a parameter vector x of length 2*n.
    #    x[0:n] are log-weights (softmaxed).
    #    x[n:2n] are support values (shifted/scaled to be positive).
    
    def decode(x, n):
        # x: (2n,)
        log_w = x[:n]
        s_raw = x[n:]
        
        # Weights: softmax
        log_w_max = np.max(log_w)
        w = np.exp(log_w - log_w_max)
        w = w / np.sum(w)
        
        # Support: We want s >= 0.
        # Use exp to ensure positivity, but we also want 0 to be a candidate.
        # Let's map s_raw to [0, inf) via exp(s_raw). 
        # However, exp can get large. 
        # Alternatively, use s = abs(s_raw) or s = np.exp(s_raw).
        # Let's use s = np.exp(s_raw) to allow a wide range.
        # To allow s=0, we might need a different mapping or just rely on optimization finding small exp values.
        # Actually, strict inequality s_i + s_j < 2 s_l - s_k.
        # If s_k is large, RHS is small. If s_i, s_j are small, LHS is small.
        # A good candidate is s = [0, small, large].
        # exp(s_raw) cannot be 0. 
        # Let's use s = np.exp(s_raw) * scale. 
        # Or better: s = np.abs(s_raw) is not smooth.
        # Let's use s = np.exp(s_raw). We can initialize s_raw with small values for 0.
        
        s = np.exp(s_raw)
        
        # Sort support and weights together to keep them aligned? 
        # The formula is symmetric in i,j and k,l separately, but not fully symmetric in all 4.
        # However, the set of values matters, not the order. 
        # Sorting helps with numerical stability of searchsorted? No, searchsorted handles unsorted input by sorting internally.
        # But for the measure definition, the pairs are (s_i, w_i).
        # We should return sorted support and corresponding weights.
        
        sort_idx = np.argsort(s)
        s_sorted = s[sort_idx]
        w_sorted = w[sort_idx]
        
        return s_sorted, w_sorted

    def objective(x, n):
        s, w = decode(x, n)
        # Check validity roughly
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return 1.0 # Maximize P, so minimize -P. Return large value for invalid.
        P = _compute_P_fast(s, w)
        return -P

    best_P_global = -1.0
    best_support_global = None
    best_weights_global = None
    
    # Fallback: simple baseline
    n_base = 3
    s_base = np.array([0.0, 1.0, 10.0])
    w_base = np.array([0.5, 0.25, 0.25])
    P_base = _compute_P_fast(s_base, w_base)
    best_P_global = P_base
    best_support_global = s_base.copy()
    best_weights_global = w_base.copy()
    
    # DE Settings
    from scipy.optimize import differential_evolution
    
    n_candidates = [3, 4, 5, 6]
    
    for n in n_candidates:
        if time.time() > deadline - 30:
            break
            
        # Bounds for x
        # log_w: [-10, 10]
        # s_raw: [-10, 5] (exp(-10) ~ 0, exp(5) ~ 148)
        # We might want larger support values. Let's allow s_raw up to 10 (exp(10) ~ 22000).
        # But large values might cause precision issues or just not help.
        # Let's try bounds [-5, 8] for s_raw.
        bounds = [(-10, 10)] * n + [(-5, 8)] * n
        
        # Initial population
        # We want to encourage diverse starts.
        # One start: all log_w = 0 (equal weights), s_raw = log([0.1, 1, 10, 100])
        # Another: bimodal.
        
        # DE handles initialization internally if not provided, but we can provide a seed.
        # Let's just run DE with default initialization but multiple restarts if time permits.
        
        try:
            # Use DE with maxiter limited to ensure we finish in time
            # n=3: 6 vars. n=6: 12 vars.
            # Population size = 15 * len(x)
            # For n=6, pop = 180. 
            # maxiter = 100?
            
            res = differential_evolution(
                objective, 
                bounds, 
                args=(n,),
                maxiter=100, 
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True # Use L-BFGS-B at the end for fine tuning
            )
            
            if res.success or res.fun < 0: # res.fun is -P, so < 0 means P > 0
                P_val = -res.fun
                if P_val > best_P_global:
                    s_opt, w_opt = decode(res.x, n)
                    # Re-compute P to be sure
                    P_check = _compute_P_fast(s_opt, w_opt)
                    if P_check > best_P_global:
                        best_P_global = P_check
                        best_support_global = s_opt.copy()
                        best_weights_global = w_opt.copy()
                        
        except Exception as e:
            print(f"DE failed for n={n}: {e}")
            continue
            
    # If we have time, do a local refinement on the best found so far
    if time.time() < deadline - 10 and best_support_global is not None:
        n_best = len(best_support_global)
        # Convert back to x-space for local optimization
        # s = exp(s_raw) => s_raw = log(s)
        # w = softmax(log_w) => log_w = log(w) (up to a constant, but softmax is invariant)
        
        s_cur = best_support_global
        w_cur = best_weights_global
        
        # Avoid log(0)
        s_safe = np.maximum(s_cur, 1e-10)
        w_safe = np.maximum(w_cur, 1e-10)
        
        x0 = np.concatenate([np.log(w_safe), np.log(s_safe)])
        
        from scipy.optimize import minimize
        
        def neg_P_local(x, n):
            s, w = decode(x, n)
            if np.any(s < -1e-9) or np.any(w < -1e-9):
                return 1.0
            P = _compute_P_fast(s, w)
            return -P
            
        try:
            res_local = minimize(
                neg_P_local, 
                x0, 
                args=(n_best,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            P_local = -res_local.fun
            if P_local > best_P_global:
                s_opt, w_opt = decode(res_local.x, n_best)
                P_check = _compute_P_fast(s_opt, w_opt)
                if P_check > best_P_global:
                    best_P_global = P_check
                    best_support_global = s_opt.copy()
                    best_weights_global = w_opt.copy()
        except Exception:
            pass
            
    # Final Validation and Cleanup
    if best_support_global is None or len(best_support_global) == 0:
        return best_support_global, best_weights_global
        
    # Ensure strict validity
    best_support_global = np.maximum(best_support_global, 0.0)
    best_weights_global = np.maximum(best_weights_global, 0.0)
    
    # Normalize weights
    w_sum = np.sum(best_weights_global)
    if w_sum <= 1e-9:
        return best_support_global, best_weights_global
        
    best_weights_global = best_weights_global / w_sum
    
    # Check for NaN/Inf
    if not (np.all(np.isfinite(best_support_global)) and np.all(np.isfinite(best_weights_global))):
        return best_support_global, best_weights_global
        
    return best_support_global, best_weights_global

# EVOLVE-BLOCK-END

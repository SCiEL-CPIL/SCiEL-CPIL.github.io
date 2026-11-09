# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
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
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get CDF values
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    # Sum over k, l with weights w_k w_l
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    
    return float(P)

def _decode_params(log_w, log_s):
    """
    Decode raw parameters to valid support and weights.
    log_w: raw weights (softmaxed)
    log_s: raw support (exponentiated to ensure > 0, but we allow 0 by mapping small negatives to 0? 
           Actually, strict constraint s_i >= 0. 
           Using exp(log_s) ensures s_i > 0. If we need s_i=0, we can't reach it with exp.
           However, the optimal 2-point solution has s_1=0. 
           To allow 0, we can use a softplus or just clip. 
           Let's use s = np.maximum(np.exp(log_s) - 1.0, 0.0)? No, that's messy.
           
           Alternative: The problem allows s_i >= 0. 
           If we use s = np.exp(log_s), we are restricted to s > 0. 
           But the limit s->0 is valid. 
           Let's check if the optimal is exactly 0 or close to 0.
           For the 2-point case (0, L), P = alpha * (1-alpha) * Prob[X1+X2+X3 < 2L - 0] + ...
           Actually, if s_1=0, it's a boundary solution. 
           To handle boundary solutions, we can use a transformation that maps R to [0, inf).
           s = np.exp(log_s) is fine if we initialize log_s with very negative numbers for 0.
           But exp(-100) is 0 in float64. So s=0 is reachable numerically.
           """
    # Softmax for weights
    log_w_max = np.max(log_w)
    w = np.exp(log_w - log_w_max)
    w = w / np.sum(w)
    
    # Support
    s = np.exp(log_s)
    # Clip to 0 to handle underflow which effectively means 0
    s = np.maximum(s, 0.0)
    
    # Sort to ensure consistent ordering (though mathematically order doesn't matter for P, 
    # it helps with numerical stability and debugging)
    sort_idx = np.argsort(s)
    s_sorted = s[sort_idx]
    w_sorted = w[sort_idx]
    
    return s_sorted, w_sorted

def _objective(log_w, log_s):
    s, w = _decode_params(log_w, log_s)
    P = _compute_P_fast(s, w)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. 
    Uses Structured Grid Search + Coordinate Descent.
    """
    t_start = time.time()
    deadline = t_start + 600.0
    
    best_P_global = -1.0
    best_support_global = None
    best_weights_global = None
    
    # Fallback: simple baseline
    s_base = np.array([0.0, 1.0, 10.0])
    w_base = np.array([0.5, 0.25, 0.25])
    P_base = _compute_P_fast(s_base, w_base)
    best_P_global = P_base
    best_support_global = s_base.copy()
    best_weights_global = w_base.copy()
    
    def update_best(s, w, P_val):
        nonlocal best_P_global, best_support_global, best_weights_global
        if P_val > best_P_global:
            best_P_global = P_val
            best_support_global = s.copy()
            best_weights_global = w.copy()

    # --- Phase 1: Analytic/Coarse Grid Search for 2-point and 3-point solutions ---
    
    # 2-Point Solution: s = [0, L], w = [1-alpha, alpha]
    # P = Prob[X1+X2+X3 < 2X4]
    # X4 can be 0 or L.
    # If X4=0 (prob 1-alpha): 2X4=0. Need X1+X2+X3 < 0. Since X>=0, this is impossible (strict inequality). P=0.
    # If X4=L (prob alpha): 2X4=2L. Need X1+X2+X3 < 2L.
    # X1,X2,X3 are 0 or L.
    # Sum < 2L means sum is 0 or L.
    # Sum=0: all 0. Prob (1-alpha)^3.
    # Sum=L: one L, two 0s. Prob 3 * alpha * (1-alpha)^2.
    # Sum=2L: two Ls, one 0. Prob 3 * alpha^2 * (1-alpha). (Not < 2L)
    # Sum=3L: three Ls. Prob alpha^3. (Not < 2L)
    # So P = alpha * [ (1-alpha)^3 + 3*alpha*(1-alpha)^2 ]
    # Let's maximize f(alpha) = alpha * (1-alpha)^2 * (1-alpha + 3alpha) = alpha * (1-alpha)^2 * (1+2alpha)
    
    alphas = np.linspace(0.01, 0.99, 100)
    for a in alphas:
        P_2pt = a * (1-a)**2 * (1 + 2*a)
        if P_2pt > best_P_global:
            # Construct solution
            # s = [0, L]. L can be anything? 
            # The condition depends on relative values. If s=[0, L], the "2L" threshold is relative to L.
            # If we scale L, the condition X1+X2+X3 < 2X4 is scale invariant?
            # X_i are scaled by L. LHS ~ L, RHS ~ L. Yes, scale invariant.
            # So L=1 is fine.
            s_opt = np.array([0.0, 1.0])
            w_opt = np.array([1-a, a])
            P_check = _compute_P_fast(s_opt, w_opt)
            update_best(s_opt, w_opt, P_check)
            
    # 3-Point Solution: s = [0, A, B], w = [w0, wA, wB]
    # We can do a coarse grid search here.
    # To save time, we'll only run this if we have time.
    if time.time() < deadline - 100:
        # Grid for A, B, wA, wB (w0 = 1-wA-wB)
        # Keep it small.
        A_vals = np.linspace(0.1, 2.0, 5)
        B_vals = np.linspace(2.0, 10.0, 5)
        # Weights: wA, wB. w0 = 1-wA-wB > 0.
        # Let's sample wA, wB in a triangle.
        wA_vals = np.linspace(0.05, 0.5, 5)
        wB_vals = np.linspace(0.05, 0.5, 5)
        
        for A in A_vals:
            for B in B_vals:
                if B <= A: continue
                for wA in wA_vals:
                    for wB in wB_vals:
                        if wA + wB >= 1.0: continue
                        w0 = 1.0 - wA - wB
                        s_opt = np.array([0.0, A, B])
                        w_opt = np.array([w0, wA, wB])
                        P_val = _compute_P_fast(s_opt, w_opt)
                        update_best(s_opt, w_opt, P_val)

    # --- Phase 2: Local Coordinate Descent ---
    # Start from the best solution found so far.
    # Represent as log_w and log_s.
    
    def init_params_from_solution(s, w):
        # Handle zeros in s and w
        s_safe = np.maximum(s, 1e-10)
        w_safe = np.maximum(w, 1e-10)
        log_s = np.log(s_safe)
        log_w = np.log(w_safe)
        return log_w, log_s

    def coordinate_descent(log_w, log_s, n, time_limit=60.0):
        """
        Perform cyclic coordinate descent on log_w and log_s.
        """
        t_cd_start = time.time()
        current_P = -_objective(log_w, log_s)
        
        # Number of steps
        max_steps = 200
        
        for step in range(max_steps):
            if time.time() > t_cd_start + time_limit:
                break
            
            # Optimize log_w (weights) with log_s fixed
            # We can use scipy minimize for 1D or simple line search.
            # Since n is small (<=4), we can just do a simple grid search or scipy minimize.
            # Let's use scipy minimize on the vector log_w.
            from scipy.optimize import minimize_scalar, minimize
            
            # Optimize each log_w_i? No, they are coupled via softmax.
            # Better: Optimize the vector log_w directly using L-BFGS-B (fast for small n).
            # Or even simpler: Since softmax is smooth, we can use minimize.
            
            res_w = minimize(
                lambda x: _objective(x, log_s),
                log_w,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            if res_w.fun < _objective(log_w, log_s): # If improved
                log_w = res_w.x
            
            # Optimize log_s (support) with log_w fixed
            res_s = minimize(
                lambda x: _objective(log_w, x),
                log_s,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            if res_s.fun < _objective(log_w, log_s): # If improved
                log_s = res_s.x
                
            # Check for convergence
            new_P = -_objective(log_w, log_s)
            if abs(new_P - current_P) < 1e-10:
                break
            current_P = new_P
            
        s_final, w_final = _decode_params(log_w, log_s)
        P_final = _compute_P_fast(s_final, w_final)
        return s_final, w_final, P_final

    # Apply coordinate descent to the best solution
    if best_support_global is not None and time.time() < deadline - 30:
        n_best = len(best_support_global)
        log_w0, log_s0 = init_params_from_solution(best_support_global, best_weights_global)
        
        # Run CD with a tight time limit
        s_cd, w_cd, P_cd = coordinate_descent(log_w0, log_s0, n_best, time_limit=30.0)
        update_best(s_cd, w_cd, P_cd)
        
        # If we have time, try CD on a slightly perturbed version to escape local minima
        if time.time() < deadline - 30:
            # Perturb log_s slightly
            log_s_pert = log_s0 + np.random.normal(0, 0.1, size=n_best)
            # Ensure s > 0
            s_pert, w_pert = _decode_params(log_w0, log_s_pert)
            # Re-optimize weights for this perturbed support
            log_w_pert, _ = init_params_from_solution(s_pert, w_pert)
            
            s_cd2, w_cd2, P_cd2 = coordinate_descent(log_w_pert, log_s_pert, n_best, time_limit=30.0)
            update_best(s_cd2, w_cd2, P_cd2)

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
        # Fallback to base
        return s_base, w_base
        
    return best_support_global, best_weights_global

# EVOLVE-BLOCK-END

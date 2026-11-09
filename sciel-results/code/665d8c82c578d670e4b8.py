import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Uses vectorized operations for speed.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Pairwise sums matrix: S[i,j] = s_i + s_j
    S = support[:, None] + support[None, :]
    
    # Weights outer product: W[i,j] = w_i * w_j
    W = weights[:, None] * weights[None, :]
    
    # Flatten for easier manipulation
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    total_prob = 0.0
    for k in range(n):
        threshold = 2.0 * support[k]
        # Condition: S_flat + support[l] < threshold  =>  S_flat < threshold - support[l]
        # We need sum_{i,j,l} w_i w_j w_l * I(s_i+s_j+s_l < 2*s_k)
        # Let T = S_flat (size n^2). We need for each l, sum_{i,j} w_i w_j I(T_ij + s_l < 2*s_k)
        # This is equivalent to: sum_l w_l * (sum_{i,j} w_i w_j I(T_ij < 2*s_k - s_l))
        
        # For fixed k, let's compute prob3 = P[X1+X2+X3 < 2*s_k]
        # prob3 = sum_l w_l * C_l, where C_l = sum_{i,j} w_i w_j I(s_i+s_j < 2*s_k - s_l)
        
        # Instead of looping l, we can vectorize over l.
        # Let thresholds_l = 2*s_k - s_l (vector of length n)
        # For each l, we need the sum of W_flat where S_flat < thresholds_l[l].
        
        # Since S_flat is sorted? No, S_flat is not sorted.
        # We can sort S_flat and use searchsorted for each l.
        # But doing this for each k and each l might still be heavy if n is large, but n is small (<=6).
        # n^2 <= 36. Sorting 36 elements is trivial.
        
        # Let's pre-sort S_flat and W_flat? No, W_flat corresponds to specific (i,j).
        # We need to sum W_flat[indices] where S_flat[indices] < val.
        # This is a weighted prefix sum problem.
        
        # Sort S_flat
        sort_idx = np.argsort(S_flat)
        S_sorted = S_flat[sort_idx]
        W_sorted = W_flat[sort_idx]
        W_cumsum = np.cumsum(W_sorted)
        
        # For each l, find count of S_sorted < (2*s_k - s_l)
        # np.searchsorted(S_sorted, val, side='left') gives index of first element >= val.
        # So elements [0:index] are < val.
        
        thresholds_l = 2.0 * support[k] - support
        # Get indices
        indices = np.searchsorted(S_sorted, thresholds_l, side='left')
        
        # Sum of weights for those indices
        # If index is 0, sum is 0. Else sum is W_cumsum[index-1].
        # Be careful with indices == 0.
        # We can use np.take with padding or just handle it.
        
        # Efficient way:
        # Create an array of sums for each l.
        # sums = np.where(indices > 0, W_cumsum[indices - 1], 0.0)
        # But indices-1 might be negative.
        
        # Let's use a safe approach:
        # Create a padded cumsum array: [0, W_cumsum[0], W_cumsum[1], ...]
        W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
        
        # Now sum for l is W_cumsum_padded[indices[l]]
        # Because if indices[l]=0, we want 0. W_cumsum_padded[0]=0.
        # If indices[l]=1, we want W_cumsum[0]. W_cumsum_padded[1]=W_cumsum[0]. Correct.
        
        sums_for_l = W_cumsum_padded[indices]
        
        # prob3 = sum_l w_l * sums_for_l
        prob3 = np.dot(weights, sums_for_l)
        
        total_prob += weights[k] * prob3
        
    return total_prob

def _eval_measure(support, weights):
    """
    Evaluate the probability for a given support and weights.
    Returns -P for minimization.
    """
    if len(support) != len(weights):
        return -0.5
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return -0.5
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return -0.5
    if np.sum(weights) <= 1e-9:
        return -0.5
    
    # Normalize weights for internal calculation? 
    # The metric uses w_i = weights[i] / sum(weights).
    # So we should normalize.
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return -0.5
    norm_weights = weights / w_sum
    
    p = _calc_prob_fast(support, norm_weights)
    return -p

def _params_to_measure(params, n_points):
    """
    Convert parameters to support and weights.
    params: [log_s1, ..., log_s(n-1), w1, ..., w(n-1)]
    s0 is fixed at 0.
    w0 = 1 - sum(w1..wn-1)
    """
    n = n_points
    log_s = params[:n-1]
    w_rest = params[n-1:]
    
    s = np.zeros(n)
    s[1:] = np.exp(log_s)
    
    w = np.zeros(n)
    w[1:] = w_rest
    w[0] = 1.0 - np.sum(w_rest)
    
    # Check validity
    if np.any(w < -1e-9):
        return None
    if not np.all(np.diff(s) > 0): # Strictly increasing
        return None
    if s[0] < -1e-9:
        return None
        
    return s, w

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using fast vectorized evaluation and global search.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    def try_update(prob, support, weights):
        nonlocal best_prob, best_support, best_weights
        if prob > best_prob:
            best_prob = prob
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Predefined good starting points / seeds
    # Based on intuition: 0, small, medium, large
    # We will search over n=2,3,4,5,6
    
    # 1. Latin Hypercube Sampling for initial global exploration
    # We'll sample in the parameter space for each n.
    
    # To keep it simple and robust, we'll do a multi-start Nelder-Mead
    # with random initializations from a broad range.
    
    # Ranges for log support points (excluding 0)
    # s_i > 0. Let's try log_s in [-3, 6] (s in [0.05, 400])
    # Weights in [0, 1]
    
    # We will run multiple optimization rounds.
    
    # Helper to run Nelder-Mead on a specific n
    def run_nelder_mead(n_points, init_params):
        def objective(params):
            m = _params_to_measure(params, n_points)
            if m is None:
                return 1.0 # Penalty for invalid
            s, w = m
            # Additional check: strictly positive weights?
            # The problem allows 0 weights, but strictly positive is safer for optimization.
            # The invalid condition is w < -1e-9. So 0 is valid.
            return _eval_measure(s, w)
        
        # Bounds for Nelder-Mead are not directly supported in the same way, 
        # but we can clip or use penalty.
        # The penalty in objective handles invalid states.
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < 0: # Valid result
                m = _params_to_measure(res.x, n_points)
                if m:
                    s, w = m
                    p = -res.fun
                    return p, s, w
        except Exception:
            pass
        return -1.0, None, None

    # Generate random initializations
    def random_init(n_points):
        # Log support
        log_s = np.random.uniform(-2.0, 4.0, size=n_points-1)
        # Sort to ensure increasing support
        log_s = np.sort(log_s)
        # Weights
        w_rest = np.random.uniform(0.01, 0.5, size=n_points-1)
        # Ensure sum < 1
        s_w = np.sum(w_rest)
        if s_w > 0.9:
            w_rest = w_rest * (0.9 / s_w)
        
        params = np.concatenate([log_s, w_rest])
        return params

    # Search loop
    # We will try n=3,4,5,6
    # For each n, do several random restarts.
    
    num_restarts = 20 # Total budget for restarts across all n? Or per n?
    # Let's do 5 restarts per n.
    
    for n_points in [3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
            
        for i in range(5):
            if time.time() > deadline - 10:
                break
                
            init_params = random_init(n_points)
            p, s, w = run_nelder_mead(n_points, init_params)
            
            if p > best_prob:
                try_update(p, s, w)
                
            # Gradient polishing? Nelder-Mead doesn't use gradients.
            # We can try a local refinement using BFGS if we had gradients, 
            # but we don't easily have analytical gradients.
            # However, we can do a "fine" Nelder-Mead with tighter tolerances 
            # or a random local search.
            
            # Let's do a simple local random search around the best found for this n
            # to polish.
            if s is not None:
                # Convert back to params?
                # It's easier to just perturb the params that gave the best result.
                # But we lost the exact params in run_nelder_mead return.
                # Let's modify run_nelder_mead to return params too.
                pass

    # Refinement phase: Take the best support/weights found and try to refine them
    # by perturbing the underlying parameters.
    # Since we don't have the params for the global best easily, 
    # we can reconstruct them or just do a generic local search on the support/weights space directly.
    
    # Direct local search on support/weights is tricky due to constraints (ordering, simplex).
    # Better to keep the parameterization.
    
    # Let's do one more pass:
    # If we have a best_support, we can try to optimize around it.
    # But we need the params.
    
    # Alternative: Use the best found (s, w) to generate a good initial guess for a 
    # higher-dimensional search? No, let's stick to what we have.
    
    # Let's add a "polish" step using a finer Nelder-Mead on the best candidate's parameters.
    # To do this, we need to track the best params.
    
    # Let's re-run the search logic but tracking params.
    
    best_params = None
    
    def run_search_round(n_points, num_starts):
        nonlocal best_prob, best_support, best_weights, best_params
        
        for i in range(num_starts):
            if time.time() > deadline - 5:
                return
                
            init_params = random_init(n_points)
            
            def objective(params):
                m = _params_to_measure(params, n_points)
                if m is None:
                    return 1.0
                s, w = m
                return _eval_measure(s, w)
            
            try:
                res = minimize(objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if res.fun < 0:
                    p = -res.fun
                    if p > best_prob:
                        m = _params_to_measure(res.x, n_points)
                        if m:
                            s, w = m
                            best_prob = p
                            best_support = s
                            best_weights = w
                            best_params = res.x
            except Exception:
                pass

    # Run search rounds
    # Allocate time:
    # n=3: 5 starts
    # n=4: 10 starts
    # n=5: 5 starts
    # n=6: 5 starts
    
    run_search_round(3, 5)
    run_search_round(4, 10)
    run_search_round(5, 5)
    run_search_round(6, 5)
    
    # Polish the best found
    if best_params is not None and time.time() < deadline - 10:
        n_points = len(best_support)
        
        # Try a few perturbations of the best params
        for i in range(10):
            if time.time() > deadline - 5:
                break
                
            perturbation_scale = 0.01 * (i + 1)
            perturbed_params = best_params + np.random.normal(0, perturbation_scale, size=len(best_params))
            
            def objective(params):
                m = _params_to_measure(params, n_points)
                if m is None:
                    return 1.0
                s, w = m
                return _eval_measure(s, w)
            
            try:
                res = minimize(objective, perturbed_params, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < 0:
                    p = -res.fun
                    if p > best_prob:
                        m = _params_to_measure(res.x, n_points)
                        if m:
                            s, w = m
                            best_prob = p
                            best_support = s
                            best_weights = w
                            best_params = res.x
            except Exception:
                pass

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0, 10.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        best_prob = _calc_prob_fast(best_support, best_weights)

    # Final cleanup
    if np.any(best_weights < -1e-9) or np.sum(best_weights) <= 1e-9 or np.any(best_support < -1e-9):
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    wsum = np.sum(best_weights)
    if wsum > 1e-9:
        best_weights = best_weights / wsum
        
    best_support = np.maximum(best_support, 0.0)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))

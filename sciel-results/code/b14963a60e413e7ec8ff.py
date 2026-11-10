import numpy as np
import time
import scipy.optimize as so

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s buffer for safety
    
    def compute_score_fast(support, weights):
        """
        Fast computation of P[X1 + X2 + X3 < 2*X4].
        Uses vectorized operations and searchsorted.
        Assumes support is sorted and weights are normalized.
        """
        n = len(support)
        if n == 0:
            return 0.0
        
        # Normalize weights
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # We need P[X1 + X2 < 2*X4 - X3]
        # Let Z = X1 + X2. We need P[Z < T] where T = 2*X4 - X3.
        
        # Pre-compute pairwise sums of support and probabilities for Z = X1 + X2
        # s_i + s_j
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        # Flatten
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        # Sort the pairwise sums to allow efficient CDF lookup via searchsorted
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sum of probabilities for Z
        p_prefix = np.cumsum(p_sorted)
        
        # Now iterate over possible values of X3 (k) and X4 (l)
        # Threshold T_{k,l} = 2*s_l - s_k
        # We need sum_{k,l} p_k p_l P(Z < T_{k,l})
        
        # Vectorize over k, l
        # s_k shape (n, 1), s_l shape (1, n)
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        # For each threshold, find the number of s_sorted elements < threshold
        # np.searchsorted(a, v, side='left') returns index of first element >= v
        # So the count of elements < v is exactly this index.
        # If index is 0, prob is 0.
        # If index is len(s_sorted), prob is 1.0.
        
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        # Get the cumulative probability up to that index
        # p_prefix[idx-1] is the sum of p_sorted[0..idx-1]
        # We need to handle idx=0 (prob=0) and idx=n^2 (prob=1)
        
        # Create an array of probabilities
        # Using np.take with mode='clip' is tricky because we want 0 for idx=0.
        # Let's create an extended prefix array: [0, p_prefix[0], ..., p_prefix[-1]]
        # Then prob = extended_prefix[idxs]
        
        p_prefix_ext = np.empty(len(p_prefix) + 1, dtype=np.float64)
        p_prefix_ext[0] = 0.0
        p_prefix_ext[1:] = p_prefix
        
        # idxs are in range [0, n^2]
        # p_prefix_ext has length n^2 + 1
        # So p_prefix_ext[idxs] gives exactly the CDF value.
        
        prob_z_less = p_prefix_ext[idxs]
        
        # Total probability
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    def objective(params):
        """
        Objective function for optimization.
        params: flattened array of [support_0, ..., support_{n-1}, weights_0, ..., weights_{n-1}]
        Returns negative probability (since we minimize).
        """
        n = len(params) // 2
        support = params[:n]
        weights = params[n:]
        
        # Enforce constraints
        # Support must be non-negative and sorted
        support = np.maximum(support, 0.0)
        support = np.sort(support)
        
        # Weights must be non-negative and sum to 1
        weights = np.maximum(weights, 0.0)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 1.0 # Maximize (minimize negative)
        weights = weights / w_sum
        
        # Compute score
        score = compute_score_fast(support, weights)
        
        return -score

    def refine_with_gradient(support, weights, max_iters=100, tol=1e-6):
        """
        Use L-BFGS-B to refine the support and weights.
        """
        n = len(support)
        
        # Unflatten
        x0 = np.concatenate([support, weights])
        
        # Bounds
        # Support: [0, 10] (arbitrary upper bound, optimal likely in [0,2])
        # Weights: [0, 1]
        bounds = [(0.0, 10.0)] * n + [(0.0, 1.0)] * n
        
        # L-BFGS-B requires differentiable objective.
        # Our objective is piecewise linear/smooth-ish but has kinks from sorting/searchsorted.
        # L-BFGS-B might struggle with non-differentiable points, but often works well enough.
        # Alternatively, use Nelder-Mead which is derivative-free but slower.
        # Let's try L-BFGS-B first.
        
        try:
            res = so.minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol, 'disp': False}
            )
            return res.x
        except Exception:
            return x0

    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Strategy:
    # 1. Use Differential Evolution on a small n (e.g., n=8) to find global basin.
    # 2. Expand to larger n (e.g., n=16) by splitting points or adding new ones?
    #    Actually, it's better to run DE on the final size if possible, but DE is expensive.
    #    Let's run DE on n=10, then polish.
    
    n_de = 10
    rng = np.random.default_rng(42)
    
    # Bounds for DE
    # Support: [0, 5]
    # Weights: [0, 1]
    bounds_de = [(0.0, 5.0)] * n_de + [(0.0, 1.0)] * n_de
    
    # DE is stochastic, so we need to set maxiter and popsize carefully.
    # popsize=15, maxiter=100 -> 1500 evaluations.
    # Each eval is O(n^2 log n^2) ~ 100 log 100 ~ 700 ops. Very fast.
    
    # Check time
    if time.time() < deadline - 100:
        try:
            # DE minimizes, so we use objective which returns -P
            res_de = so.differential_evolution(
                objective, 
                bounds_de, 
                seed=42, 
                maxiter=200, 
                popsize=15, 
                tol=1e-6, 
                polish=False, # We will polish manually
                update_interval=10,
                callback=lambda x, convergence: (time.time() > deadline - 50)
            )
            
            if res_de.success or res_de.fun < 0:
                x_de = res_de.x
                support_de = x_de[:n_de]
                weights_de = x_de[n_de:]
                
                # Refine with L-BFGS-B
                x_refined = refine_with_gradient(support_de, weights_de, max_iters=50)
                support_refined = x_refined[:n_de]
                weights_refined = x_refined[n_de:]
                
                # Ensure validity
                support_refined = np.sort(np.maximum(support_refined, 0.0))
                weights_refined = np.maximum(weights_refined, 0.0)
                w_sum = np.sum(weights_refined)
                if w_sum > 1e-9:
                    weights_refined = weights_refined / w_sum
                
                score_refined = compute_score_fast(support_refined, weights_refined)
                
                if score_refined > best_score:
                    best_score = score_refined
                    best_support = support_refined.copy()
                    best_weights = weights_refined.copy()
        except Exception:
            pass
    
    # If DE didn't run or failed, try a few random restarts with local polish
    if best_support is None or time.time() < deadline - 50:
        n_local = 12
        for restart in range(5):
            if time.time() > deadline - 50:
                break
            
            # Random init
            support_init = np.sort(rng.uniform(0, 2, n_local))
            weights_init = rng.dirichlet(np.ones(n_local))
            
            x0 = np.concatenate([support_init, weights_init])
            bounds_local = [(0.0, 5.0)] * n_local + [(0.0, 1.0)] * n_local
            
            try:
                res_local = so.minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead', 
                    bounds=bounds_local, # Nelder-Mead ignores bounds, but objective handles it
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6, 'disp': False}
                )
                
                x_res = res_local.x
                support_res = x_res[:n_local]
                weights_res = x_res[n_local:]
                
                support_res = np.sort(np.maximum(support_res, 0.0))
                weights_res = np.maximum(weights_res, 0.0)
                w_sum = np.sum(weights_res)
                if w_sum > 1e-9:
                    weights_res = weights_res / w_sum
                
                score_res = compute_score_fast(support_res, weights_res)
                
                if score_res > best_score:
                    best_score = score_res
                    best_support = support_res.copy()
                    best_weights = weights_res.copy()
            except Exception:
                pass

    if best_support is None:
        # Fallback to seed-like uniform
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n

    # Clean up: ensure non-negative
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    
    # Final check
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
         n = 32
         return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n
    best_weights = best_weights / w_sum

    return best_support, best_weights

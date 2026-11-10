import numpy as np
import scipy.optimize as optimize
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def calc_prob(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        p_prefix = np.cumsum(p_sorted)
        
        # Thresholds for X3, X4
        # Condition: X1 + X2 < 2*X4 - X3
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        # Searchsorted on sorted s_sums
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        prob_z_less = np.zeros_like(idxs, dtype=np.float64)
        mask = idxs > 0
        if np.any(mask):
            prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
            
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    def objective_n(n, params):
        """
        Objective for scipy optimizer.
        params: [s_1, ..., s_{n-2}, w_1, ..., w_{n-2}]
        s_0 = 0, s_{n-1} = 1
        w_0 = 1 - sum(w_internal) - w_{n-1}? No, we optimize w_0..w_{n-1} directly?
        Better to optimize log-weights or just weights with penalty.
        Let's optimize:
        x[0:n-2] = internal support points
        x[n-2:2n-2] = internal weights (w_1..w_{n-2})
        w_0 and w_{n-1} are derived?
        Actually, it's easier to optimize all weights w_0..w_{n-1} and normalize, 
        but we have 2n-2 free variables if we fix s_0=0, s_{n-1}=1.
        
        Let's stick to:
        x[0:n-2] = s_1..s_{n-2}
        x[n-2:2n-2] = w_0..w_{n-1} (unnormalized)
        
        Check validity:
        1. s_0=0 < s_1 < ... < s_{n-2} < s_{n-1}=1
        2. w_i > 0
        """
        s_int = params[:n-2]
        w_raw = params[n-2:]
        
        # Check support ordering
        s_prev = 0.0
        valid = True
        for i in range(n-2):
            if s_int[i] <= s_prev or s_int[i] >= 1.0:
                valid = False
                break
            s_prev = s_int[i]
        if s_prev >= 1.0:
            valid = False
            
        # Check weights
        if np.any(w_raw <= 0):
            valid = False
            
        if not valid:
            return 1e6  # Penalty
            
        support = np.concatenate(([0.0], s_int, [1.0]))
        weights = w_raw / np.sum(w_raw)
        
        return -calc_prob(support, weights)

    def polish_with_minimize(support, weights, max_time=30.0):
        """
        Use scipy.optimize.minimize (Nelder-Mead) to polish.
        """
        n = len(support)
        if n < 3:
            return support, weights, calc_prob(support, weights)
            
        # Extract internal parameters
        s_int = support[1:-1]
        w_raw = weights  # We will optimize raw weights, they will be normalized
        
        x0 = np.concatenate((s_int, w_raw))
        
        # Objective wrapper for minimize
        def obj(x):
            return objective_n(n, x)
            
        # Nelder-Mead parameters
        # maxfev is the key to limit time
        # Estimate cost per function call
        # n^2 operations. For n=6, 36 ops. Very fast.
        # We can afford many iterations.
        
        t_start = time.time()
        # Use a timeout wrapper? No, minimize doesn't support timeout directly.
        # We can set maxfev based on remaining time.
        # Assume 1000 evaluations per second.
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        maxfev = int(remaining * 500) # Conservative estimate
        
        if maxfev < 100:
             return support, weights, calc_prob(support, weights)

        try:
            res = optimize.minimize(obj, x0, method='Nelder-Mead', 
                                    options={'maxfev': maxfev, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < 1e5: # Valid result
                s_int_new = res.x[:n-2]
                w_raw_new = res.x[n-2:]
                # Re-construct
                support_new = np.concatenate(([0.0], s_int_new, [1.0]))
                weights_new = w_raw_new / np.sum(w_raw_new)
                
                # Final check
                if np.all(np.diff(support_new) > 0) and np.all(weights_new > 0):
                    score_new = calc_prob(support_new, weights_new)
                    return support_new, weights_new, score_new
        except Exception:
            pass
            
        return support, weights, calc_prob(support, weights)

    def objective_de_n(n, params):
        """
        Objective for Differential Evolution.
        Same parameterization as polish.
        """
        return objective_n(n, params)

    best_score = -1.0
    best_support = None
    best_weights = None

    # Try n from 2 to 6
    for n in range(2, 7):
        if time.time() > deadline - 100: # Need time for polishing
            break
            
        if n == 2:
            # Special case: s=[0,1], w=[p, 1-p]
            def obj2(p):
                w = np.array([p, 1-p])
                if w[0] < 1e-9 or w[1] < 1e-9:
                    return 1e6
                s = np.array([0.0, 1.0])
                return -calc_prob(s, w)
            
            try:
                res = optimize.differential_evolution(obj2, bounds=[(1e-6, 1-1e-6)], seed=42, maxiter=200, polish=False)
                if res.fun < 1e5:
                    w = np.array([res.x[0], 1-res.x[0]])
                    s = np.array([0.0, 1.0])
                    sc = calc_prob(s, w)
                    if sc > best_score:
                        best_score = sc
                        best_support = s.copy()
                        best_weights = w.copy()
            except Exception:
                pass
            continue
            
        # n >= 3
        # Parameters: s_1..s_{n-2}, w_0..w_{n-1}
        # Bounds:
        # s_i in (0, 1)
        # w_i in (1e-6, 1)
        
        bounds = []
        for _ in range(n-2):
            bounds.append((1e-6, 1-1e-6))
        for _ in range(n):
            bounds.append((1e-6, 1.0))
            
        # Initial guess
        x0 = []
        # Uniform support
        for i in range(1, n-1):
            x0.append(i / (n-1))
        # Uniform weights
        for i in range(n):
            x0.append(1.0/n)
            
        try:
            res = optimize.differential_evolution(objective_de_n, bounds=bounds, args=(n,), 
                                                  seed=42, maxiter=300, tol=1e-6, polish=False, 
                                                  init='sobol')
            if res.fun < 1e5:
                s_int = res.x[:n-2]
                w_raw = res.x[n-2:]
                support = np.concatenate(([0.0], s_int, [1.0]))
                weights = w_raw / np.sum(w_raw)
                
                # Validate
                if np.all(np.diff(support) > 0) and np.all(weights > 0):
                    sc = calc_prob(support, weights)
                    if sc > best_score:
                        best_score = sc
                        best_support = support.copy()
                        best_weights = weights.copy()
        except Exception:
            pass

    # Polish the best result found
    if best_support is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            max_polish_time = min(remaining_time - 5, 60)
            best_support, best_weights, best_score = polish_with_minimize(
                best_support, best_weights, max_time=max_polish_time
            )
            
            # One more quick polish with smaller step if time permits
            if time.time() < deadline - 5:
                best_support, best_weights, best_score = polish_with_minimize(
                    best_support, best_weights, max_time=10
                )

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])

    # Final normalization and cleanup
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    sum_w = np.sum(best_weights)
    if sum_w > 1e-9:
        best_weights = best_weights / sum_w
    else:
        best_weights = np.ones(len(best_support)) / len(best_support)
        
    return best_support, best_weights

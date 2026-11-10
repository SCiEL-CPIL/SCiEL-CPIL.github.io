import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    def compute_score(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] efficiently for small n.
        n is expected to be small (3-5).
        """
        n = len(support)
        if n == 0:
            return 0.0
        
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        
        p = weights / w_sum
        
        # Precompute pairwise sums and products for Z = X1 + X2
        # s_sums[i, j] = s_i + s_j
        # p_prods[i, j] = p_i * p_j
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        # We need sum_{k,l} p_k p_l P(Z < 2 s_l - s_k)
        # P(Z < t) = sum_{i,j} p_i p_j I(s_i + s_j < t)
        
        # For small n, we can just loop or vectorize simply.
        # Let's vectorize the inner sum.
        # T[k, l] = 2 * s_l - s_k
        T = 2.0 * support[None, :] - support[:, None] # shape (n, n)
        
        # For each (k, l), we need sum over (i, j) where s_i + s_j < T[k, l] of p_i p_j
        # This is equivalent to summing p_prods over the mask (s_sums < T[k, l])
        
        # Efficient way for small n:
        # Create a 4D tensor? No, n is small, so n^4 is small.
        # But we can do it in O(n^3) or O(n^2 log n).
        
        # Let's use the sorting method which is robust and fast enough for n<10.
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        p_prefix = np.cumsum(p_sorted)
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        
        # idx[k, l] = number of pairs (i, j) such that s_i + s_j < T[k, l]
        # searchsorted returns the index of the first element >= T. 
        # Since we want strict inequality <, we use side='left' for the value T.
        # Wait, if s_i + s_j == T, it should NOT be included.
        # np.searchsorted(a, v, side='left') returns the first index where a[idx] >= v.
        # So all indices < idx have a[idx] < v.
        # This is correct for strict inequality.
        
        idx = np.searchsorted(s_sorted, T, side='left')
        
        # prob_z_less[k, l] = p_prefix_pad[idx[k, l]]
        prob_z_less = p_prefix_pad[idx]
        
        # Total probability
        # P = sum_{k, l} p_k p_l prob_z_less[k, l]
        total_prob = np.sum(p[:, None] * p[None, :] * prob_z_less)
        
        return total_prob

    def optimize_weights_for_support(support):
        """
        Given fixed support, find optimal weights.
        Uses differential_evolution on weights (simplex constraint handled by bounds and normalization? 
        DE doesn't handle simplex constraints directly. 
        Better: use minimize with SLSQP on weights, or parameterize weights with softmax.
        Softmax parameterization: w_i = exp(theta_i) / sum(exp(theta_j)).
        This ensures w_i > 0 and sum w_i = 1.
        """
        n = len(support)
        
        # Objective for weights (minimize negative score)
        def neg_score_weights(theta):
            # Convert theta to weights using softmax
            # Subtract max for numerical stability
            theta_shifted = theta - np.max(theta)
            exp_theta = np.exp(theta_shifted)
            w = exp_theta / np.sum(exp_theta)
            
            # Ensure support is valid (non-negative)
            if np.any(support < -1e-9):
                return 0.0
            if np.any(w < -1e-9): # w from softmax is always > 0
                return 0.0
                
            score = compute_score(support, w)
            return -score

        # Initial guess for theta: uniform weights -> theta = 0
        theta0 = np.zeros(n)
        
        # Bounds for theta: large enough to allow sparse distributions
        # exp(10) is huge, exp(-10) is tiny. 
        bounds = [(-10, 10)] * n
        
        # Run differential evolution for weights
        # Since n is small (3-5), DE is fast and robust.
        try:
            res = differential_evolution(
                neg_score_weights, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-6,
                polish=True
            )
            theta_opt = res.x
            theta_shifted = theta_opt - np.max(theta_opt)
            w_opt = np.exp(theta_shifted)
            w_opt = w_opt / np.sum(w_opt)
            
            score = compute_score(support, w_opt)
            return w_opt, score
            
        except Exception:
            # Fallback to uniform weights
            w_opt = np.ones(n) / n
            score = compute_score(support, w_opt)
            return w_opt, score

    def objective_support(support_params):
        """
        Objective for support optimization.
        support_params are the raw support points.
        We enforce non-negativity and ordering if needed? 
        Actually, the order of support points doesn't matter for the measure definition, 
        but for the optimizer, it's better to keep them sorted or allow any order.
        The measure is defined by the set of points and weights. 
        If we allow unsorted support, the optimizer might swap points, which is fine.
        However, we must ensure support >= 0.
        """
        support = np.array(support_params)
        
        # Enforce non-negativity
        if np.any(support < 0):
            # Penalize heavily or clip? 
            # Clipping might cause flat regions. 
            # Let's clip to 0 for the evaluation, but the optimizer should learn to stay positive.
            support = np.maximum(support, 0)
            
        # Optimize weights for this support
        w_opt, score = optimize_weights_for_support(support)
        
        return -score

    # Strategy:
    # 1. Try different small n (3, 4, 5).
    # 2. For each n, use Differential Evolution to optimize support points.
    # 3. Support points are in [0, L]. We need to choose L.
    #    Based on previous knowledge, L around 1-4 might be good. 
    #    Let's try a few L values or scale the support.
    #    Actually, the problem is scale-invariant? 
    #    If we scale support by c, X_i -> c X_i.
    #    P[X1+X2+X3 < 2X4] becomes P[c(X1'+X2'+X3') < 2c X4'] = P[X1'+X2'+X3' < 2X4'].
    #    So the probability is INVARIANT to scaling of the support!
    #    This means we can fix the scale, e.g., set max(support) = 1, or fix one point.
    #    Let's fix the support to be in [0, 1]. This simplifies the search space.
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    rng = np.random.default_rng(42)
    
    # Try n = 3, 4, 5
    for n in [3, 4, 5]:
        if time.time() > deadline - 30.0:
            break
            
        # Support bounds: [0, 1]
        # We don't enforce ordering in DE, but it might help to sort them afterwards.
        # The measure is invariant to permutation of support/weights pairs.
        # So we can just let DE find any points in [0, 1].
        
        bounds_support = [(0.0, 1.0)] * n
        
        # Initial population for DE
        # Maybe start with some structured points?
        # DE generates random points in bounds.
        
        try:
            res_support = differential_evolution(
                objective_support, 
                bounds_support, 
                seed=42, 
                maxiter=50, 
                tol=1e-5,
                popsize=10,
                polish=True,
                # Callback to check time
                callback=lambda x, convergence: (time.time() < deadline - 10.0)
            )
            
            support_opt = np.array(res_support.x)
            support_opt = np.maximum(support_opt, 0)
            
            # Get optimal weights for this support
            w_opt, score = optimize_weights_for_support(support_opt)
            
            if score > best_score:
                best_score = score
                best_support = support_opt.copy()
                best_weights = w_opt.copy()
                
        except Exception as e:
            pass
    
    # If n=3,4,5 didn't yield a good result or time ran out, try a fixed known-good distribution
    # or a simple heuristic.
    if best_support is None or best_score < 0.5:
        # Fallback: Uniform on [0, 1] with n=10
        n = 10
        best_support = np.linspace(0, 1, n)
        best_weights = np.ones(n) / n
        best_score = compute_score(best_support, best_weights)
        
    # Final cleanup
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        best_support = np.array([0.0])
        best_weights = np.array([1.0])

    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    return best_support, best_weights

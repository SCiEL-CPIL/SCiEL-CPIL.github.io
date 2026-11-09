# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _prob_brute(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] by direct enumeration for small n."""
    n = len(support)
    w = weights / weights.sum()
    
    total_prob = 0.0
    for idx4 in range(n):
        w4 = w[idx4]
        threshold = 2.0 * support[idx4]
        p_sum = 0.0
        for idx1 in range(n):
            for idx2 in range(n):
                for idx3 in range(n):
                    if support[idx1] + support[idx2] + support[idx3] < threshold:
                        p_sum += w[idx1] * w[idx2] * w[idx3]
        total_prob += w4 * p_sum
    
    return total_prob

def _prob_3point_analytic(p, q, M):
    """
    Compute P[X1+X2+X3 < 2*X4] for 3-point distribution at {0, 1, M}.
    p = P(X=0), q = P(X=1), r = P(X=M) = 1-p-q
    """
    r = 1.0 - p - q
    if r < -1e-12:
        return 0.0
    r = max(r, 0.0)
    
    if M > 2.0 + 1e-9:
        # X4=0: P(sum<0) = 0
        # X4=1: P(sum<2) = p^3 + 3*p^2*q
        prob_x4_1 = p**3 + 3*p**2*q
        
        # X4=M: P(sum<2M). For M>2, sums < 2M are:
        # From {0,1}: all sums 0,1,2,3 are < M < 2M. Prob (p+q)^3 = (1-r)^3
        # From {M,0,1}: sums M, M+1, M+2, M+3 are < 2M. Prob r(p+q)^2 = r(1-r)^2
        # Total: (1-r)^3 + r(1-r)^2 = (1-r)^2
        prob_x4_M = (1.0 - r)**2
        
        total = q * prob_x4_1 + r * prob_x4_M
        return total
    else:
        # Fallback to brute force for M <= 2
        support = np.array([0.0, 1.0, M])
        weights = np.array([p, q, r])
        return _prob_brute(support, weights)

def _optimize_3point():
    """Optimize the 3-point distribution {0, 1, M} using fine grid + local refinement."""
    
    best_p, best_q, best_M, best_prob = 0.5, 0.3, 10.0, 0.0
    
    # For M > 2, P = q*(p^3 + 3*p^2*q) + r*(1-r)^2
    # where r = 1-p-q
    # We maximize over p, q with p+q < 1.
    
    # Do a fine grid search over p and q
    n_grid = 300
    p_vals = np.linspace(0.0, 1.0, n_grid)
    q_vals = np.linspace(0.0, 1.0, n_grid)
    
    best_prob_grid = 0.0
    best_p_grid, best_q_grid = 0.5, 0.3
    
    for p in p_vals:
        for q in q_vals:
            if p + q >= 1.0:
                continue
            r = 1.0 - p - q
            if r < 0:
                continue
            # Use M=10 (any M>2 works)
            prob = _prob_3point_analytic(p, q, 10.0)
            if prob > best_prob_grid:
                best_prob_grid = prob
                best_p_grid, best_q_grid = p, q
    
    best_p, best_q, best_prob = best_p_grid, best_q_grid, best_prob_grid
    best_M = 10.0  # Any M > 2 gives same P
    
    # Refine with local optimization in (p, q) space
    def neg_prob_pq(params):
        p, q = params
        if p < 0 or q < 0 or p + q >= 1:
            return 1.0
        return -_prob_3point_analytic(p, q, 10.0)
    
    try:
        res_local = minimize(
            neg_prob_pq, [best_p, best_q],
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14}
        )
        if res_local.fun < -best_prob:
            best_p, best_q = res_local.x
            best_prob = -res_local.fun
    except Exception:
        pass
    
    # Try multiple starting points for local optimization
    start_points = [
        (0.3, 0.3), (0.4, 0.2), (0.5, 0.1), (0.2, 0.4),
        (0.6, 0.1), (0.1, 0.5), (0.5, 0.3), (0.3, 0.5),
        (0.4, 0.4), (0.2, 0.2), (0.6, 0.2), (0.2, 0.6),
    ]
    for sp in start_points:
        try:
            res = minimize(
                neg_prob_pq, sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res.fun < -best_prob:
                best_p, best_q = res_local.x
                best_prob = -res_local.fun
        except Exception:
            pass
    
    best_M = 10.0  # Any M > 2
    return best_p, best_q, best_M, best_prob

def _optimize_4point_fixed_support(support):
    """Optimize weights for a fixed 4-point support."""
    n = len(support)
    
    def prob_from_weights(weights):
        w = np.array(weights)
        if np.any(w < 0):
            return 0.0
        total = w.sum()
        if total <= 0:
            return 0.0
        w = w / total
        return _prob_brute(support, w)
    
    def neg_prob(weights):
        return -prob_from_weights(weights)
    
    # Simple grid + local refine for weights (3 params due to sum=1)
    # Or use differential evolution
    bounds = [(0, 1)] * n
    
    try:
        result = differential_evolution(
            neg_prob, bounds,
            maxiter=100, popsize=15,
            tol=1e-8, seed=42,
            polish=True
        )
        best_weights = result.x
        best_prob = -result.fun
        # Normalize
        best_weights = best_weights / best_weights.sum()
    except Exception:
        best_weights = np.ones(n) / n
        best_prob = prob_from_weights(best_weights)
        
    return support, best_weights, best_prob

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin
    
    best_support = None
    best_weights = None
    best_prob = -1.0
    
    # 1. Try 3-point optimization (fast and robust)
    try:
        p, q, M, prob_3 = _optimize_3point()
        elapsed = time.time() - start_time
        
        support_3 = np.array([0.0, 1.0, M])
        weights_3 = np.array([p, q, 1.0 - p - q])
        weights_3 = np.maximum(weights_3, 0)
        if weights_3.sum() > 1e-9:
            weights_3 = weights_3 / weights_3.sum()
            
        # Verify with brute force
        true_prob_3 = _prob_brute(support_3, weights_3)
        
        if true_prob_3 > best_prob:
            best_prob = true_prob_3
            best_support = support_3
            best_weights = weights_3
    except Exception as e:
        pass
    
    # 2. Try 4-point optimizations with fixed supports
    # Support A: {0, 1, 2, 10}
    if time.time() < deadline:
        try:
            sup, wts, prob = _optimize_4point_fixed_support(np.array([0.0, 1.0, 2.0, 10.0]))
            if prob > best_prob:
                best_prob = prob
                best_support = sup
                best_weights = wts
        except Exception:
            pass
            
    # Support B: {0, 1, 5, 10}
    if time.time() < deadline:
        try:
            sup, wts, prob = _optimize_4point_fixed_support(np.array([0.0, 1.0, 5.0, 10.0]))
            if prob > best_prob:
                best_prob = prob
                best_support = sup
                best_weights = wts
        except Exception:
            pass

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.4, 0.3, 0.3])
        best_weights = best_weights / best_weights.sum()

    # Final validation
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.4, 0.3, 0.3])
        best_weights = best_weights / best_weights.sum()
    if np.sum(best_weights) <= 1e-9:
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.4, 0.3, 0.3])
        best_weights = best_weights / best_weights.sum()

    return best_support, best_weights

# EVOLVE-BLOCK-END

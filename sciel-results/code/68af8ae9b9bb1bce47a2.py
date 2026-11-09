"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d) using efficient Pareto-frontier hypervolume search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses a Pareto-frontier approach over (cost, -expected_improvement) to select
    candidates that maximize hypervolume contribution relative to the current
    non-dominated set. For efficiency, it uses exact marginal HV calculation for
    small Pareto fronts and a fast vectorized proxy for large fronts.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    n = len(mu)
    
    # Compute expected improvement (EI) for each candidate
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Objectives: minimize cost, minimize -EI (i.e., maximize EI)
    obj_cost = cost.copy()
    obj_neg_ei = -ei.copy()
    
    # Find the non-dominated set
    indices = np.argsort(obj_cost)
    sorted_ei = ei[indices]
    
    is_nondominated = np.ones(n, dtype=bool)
    max_ei_so_far = -np.inf
    for i in range(n):
        idx = indices[i]
        if sorted_ei[i] <= max_ei_so_far:
            is_nondominated[idx] = False
        else:
            max_ei_so_far = sorted_ei[i]
    
    nd_indices = np.where(is_nondominated)[0]
    nd_cost = obj_cost[nd_indices]
    nd_ei = ei[nd_indices]
    
    # Reference point for HV
    ref_cost = np.max(obj_cost) * 1.1
    ref_neg_ei = np.min(obj_neg_ei) - 0.1
    
    scores = np.zeros(n)
    
    K = len(nd_cost)
    
    # Threshold for switching to fast approximation
    # If K is large, the O(N*K) loop is too slow.
    # Use a vectorized proxy: EI / (cost + epsilon) which correlates with HV contribution efficiency.
    if K > 100:
        # Fast proxy: Cost-Effective EI
        # This prioritizes high EI for low cost, effectively navigating the Pareto front
        scores = ei / (obj_cost + 1e-6)
    else:
        # Exact Marginal HV Calculation
        if K > 0:
            # Sort non-dominated points by cost
            nd_order = np.argsort(nd_cost)
            front_c = nd_cost[nd_order]
            front_ne = obj_neg_ei[nd_indices[nd_order]]
            
            for i in range(n):
                c_i = obj_cost[i]
                ne_i = obj_neg_ei[i]
                
                if c_i >= ref_cost:
                    scores[i] = 0.0
                    continue
                
                integral = 0.0
                
                # Part before the first front point
                if c_i < front_c[0]:
                    length = front_c[0] - c_i
                    height = ref_neg_ei - ne_i
                    if height > 0:
                        integral += length * height
                
                # Part within the front intervals
                for k in range(K):
                    interval_start = max(c_i, front_c[k])
                    interval_end = min(ref_cost, front_c[k + 1] if k + 1 < K else ref_cost)
                    
                    if interval_start >= interval_end:
                        continue
                    
                    height = front_ne[k] - ne_i
                    if height > 0:
                        integral += (interval_end - interval_start) * height
                
                scores[i] = integral
        else:
            scores = ei / (obj_cost + 1e-6)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

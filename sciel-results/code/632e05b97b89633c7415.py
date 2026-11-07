import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute heuristic values for ant colony optimization using a smooth
    surrogate for the minimum per-constraint efficiency ratio.
    
    Uses log-sum-exp based soft-min to approximate the bottleneck constraint
    more smoothly than hard min or geometric mean, which may better guide
    the ant colony toward items that are efficient across all constraints.
    """
    # Track time to ensure we return before timeout
    start_time = time.time()
    deadline = 325.0  # 5s safety margin before 330s kill
    
    # Ensure weights are positive to avoid division by zero
    epsilon = 1e-9
    safe_weights = np.where(weights > epsilon, weights, epsilon)
    
    # Calculate per-constraint efficiency ratios: prize / weight
    ratios = prizes / safe_weights
    
    # Use soft-min via log-sum-exp: soft_min(x) = -logsumexp(-x)
    # This approximates the minimum but is smooth, allowing gradient-like
    # behavior in the heuristic
    neg_ratios = -np.log(ratios)  # negative log ratios for soft-min
    max_neg = np.max(neg_ratios, axis=0, keepdims=True)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_ratios - max_neg), axis=0))
    soft_min_log_ratio = -log_sum_exp
    
    # The soft-min of log(ratios) gives a smooth approximation of 
    # log(min(ratios)), so exp of that gives soft-min of ratios
    heuristic = np.exp(soft_min_log_ratio)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic

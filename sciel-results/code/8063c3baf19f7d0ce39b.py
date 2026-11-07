import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute heuristic values for ant colony optimization using a soft-min
    surrogate for the minimum per-constraint efficiency ratio.
    
    This uses a log-sum-exp based soft-min to approximate the bottleneck
    constraint more smoothly than hard min or geometric mean, which may
    better guide the ant colony toward items that are efficient across
    all constraints.
    """
    # Track time to ensure we return before timeout
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s limit
    
    # Ensure weights are positive to avoid division by zero
    epsilon = 1e-9
    safe_weights = np.where(weights > epsilon, weights, epsilon)
    
    # Calculate per-constraint efficiency ratios: prize / weight
    ratios = prizes / safe_weights
    
    # Calculate log of ratios
    log_ratios = np.log(ratios)
    
    # Use soft-min via log-sum-exp:
    # soft_min(x, alpha) = -1/alpha * log(sum(exp(-alpha * x)))
    # As alpha -> inf, this approaches the hard min.
    # We use a moderate alpha to get a smooth approximation.
    alpha = 10.0
    neg_alpha_log_ratios = -alpha * log_ratios
    
    # For numerical stability, subtract the max before exp
    max_neg_alpha = np.max(neg_alpha_log_ratios, axis=0, keepdims=True)
    exp_terms = np.exp(neg_alpha_log_ratios - max_neg_alpha)
    
    # Sum over constraints (axis=0)
    sum_exp = np.sum(exp_terms, axis=0)
    
    # soft_min = -1/alpha * (max_neg_alpha[0] + log(sum_exp))
    # Note: max_neg_alpha is shape (1, 100), sum_exp is shape (100,)
    log_sum = np.log(sum_exp)
    soft_min_log = -1.0 / alpha * (max_neg_alpha[0] + log_sum)
    
    heuristic = np.exp(soft_min_log)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic

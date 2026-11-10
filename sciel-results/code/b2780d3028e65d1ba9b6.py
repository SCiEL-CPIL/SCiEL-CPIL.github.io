import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard (though calculation is fast, good practice per directive)
    t0 = time.time()
    deadline = 300.0
    if time.time() - t0 > deadline:
        # Fallback to simple ratios if we somehow are slow (unlikely)
        pass

    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    # Per-constraint ratios
    ratios = prizes / weights  # shape (5, 100)
    
    # Soft-min via log-sum-exp
    # We want to approximate min_j(ratios[j, i])
    # softmin(x) = -1/beta * log(sum(exp(-beta * x)))
    beta = 10.0
    
    # Numerical stability: subtract max
    neg_beta_ratios = -beta * ratios
    max_neg = neg_beta_ratios.max(axis=0)
    
    # If max_neg is -inf (shouldn't happen with 1e-9), handle it
    # exp(neg_beta_ratios - max_neg)
    exp_terms = np.exp(neg_beta_ratios - max_neg)
    sum_exp = exp_terms.sum(axis=0)
    
    # log_sum_exp = max_neg + log(sum_exp)
    log_sum_exp = max_neg + np.log(sum_exp)
    
    soft_min = -1.0 / beta * log_sum_exp
    
    # Slack bonus: items with smaller total weight are more flexible
    # Using sum of weights as a proxy for total resource consumption
    total_weight = weights.sum(axis=0)
    slack = 1.0 / (total_weight + 1e-9)
    
    # Combine
    heuristic = soft_min * slack
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic

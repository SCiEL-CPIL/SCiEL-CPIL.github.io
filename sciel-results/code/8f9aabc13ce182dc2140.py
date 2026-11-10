"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure shapes are (5, 100)
    n_constraints, n_items = prizes.shape
    
    # Ensure positive values
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    # Per-constraint value density
    ratios = prizes / weights  # shape (5, 100)
    
    # Soft-min across constraints using log-sum-exp trick
    # softmin(x_1, ..., x_k) = -logsumexp(-x_1, ..., -x_k)
    # We want items that have good ratio in ALL constraints
    neg_ratios = -ratios
    max_neg = np.max(neg_ratios, axis=0, keepdims=True)
    log_sum_exp = max_neg.squeeze(0) + np.log(np.sum(np.exp(neg_ratios - max_neg), axis=0))
    soft_min = -log_sum_exp  # shape (100,)
    
    # Total weight per item (sum across constraints)
    total_weights = weights.sum(axis=0)  # shape (100,)
    
    # Slack factor: items with lower total weight are more attractive
    # Because they leave more room in all constraints
    # Normalize: min total weight gets highest slack bonus
    min_tw = np.min(total_weights)
    max_tw = np.max(total_weights)
    if max_tw > min_tw:
        slack_factor = 1.0 + 0.5 * (1.0 - (total_weights - min_tw) / (max_tw - min_tw))
    else:
        slack_factor = np.ones(n_items)
    
    # Combine: bottleneck efficiency * slack factor
    # This strongly prioritizes items that are efficient in the tightest constraint
    # AND leave room for other items
    scores = soft_min * slack_factor
    
    # Ensure all positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

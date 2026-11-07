"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using a smooth surrogate for the multi-constraint bottleneck.
    Uses log-sum-exp to create a smooth max over constraint weights, penalizing items that
    are heavy in any single constraint more than a simple sum would, but more smoothly
    than a hard max.
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # Smooth max of weights across constraints using log-sum-exp
    # This approximates max(weights) but is differentiable and smoother
    # Using a moderate temperature to balance between sum and max
    max_w = weights.max(axis=0)
    # Normalize weights for numerical stability in log-sum-exp
    normalized_w = weights / (max_w + 1e-9)
    # log-sum-exp: log(sum(exp(w_i))) ≈ max(w_i) + log(sum(exp(w_i - max(w_i))))
    # We want a measure that is close to max but accounts for all constraints
    # Use temperature T=1 for direct log-sum-exp
    log_sum_exp = max_w + np.log(np.sum(np.exp(normalized_w - 1), axis=0))
    
    # Combine: use the smooth max as the cost measure
    # This is more sensitive to individual constraint violations than sum
    cost = log_sum_exp + 1e-9
    
    # Heuristic: prize density with smooth bottleneck cost
    h = total_prize / cost
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    # Normalize to [0, 1] range to be comparable with pheromones starting at 1
    h_min = h.min()
    h_max = h.max()
    if h_max > h_min:
        h = (h - h_min) / (h_max - h_min)
    else:
        h = np.ones_like(h)
    
    # Scale to be in a reasonable range (similar magnitude to pheromones)
    # Pheromones start at 1, so we want heuristics to be O(1)
    return h
# EVOLVE-BLOCK-END

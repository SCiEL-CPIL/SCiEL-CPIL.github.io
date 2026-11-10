"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of the
    multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    t0 = time.time()
    DEADLINE = 295.0  # safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure correct shape
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback to simple heuristic if shape is wrong
        return np.ones(100)
    
    # Ensure positive values
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    # Per-constraint efficiency: prize/weight for each constraint
    ratios = prizes / weights  # shape (5, 100)
    
    # We want items that have good prize-to-weight ratio across ALL constraints.
    # The binding constraint is the one with the worst ratio for each item.
    # So we use the minimum ratio across constraints as a base score.
    
    # Use a soft-min to smoothly handle the minimum across constraints
    # softmin(x, beta) = -logsumexp(-beta * x) / beta
    # For beta -> infinity, this approaches min(x)
    
    # Try different beta values to find a good balance
    # Higher beta means closer to true min, lower beta is smoother
    
    # Let's use a geometric mean approach as well, which is equivalent to
    # the product of ratios raised to 1/n
    # log(geometric_mean) = mean(log(ratios))
    
    log_ratios = np.log(ratios)  # shape (5, 100)
    
    # Geometric mean of ratios across constraints
    log_geo_mean = np.mean(log_ratios, axis=0)  # shape (100,)
    geo_mean = np.exp(log_geo_mean)
    
    # Also compute the min ratio
    min_ratio = np.min(ratios, axis=0)  # shape (100,)
    
    # Combine: use a weighted combination of geometric mean and min ratio
    # Items with high values on both are preferred
    # The geometric mean rewards balanced performance
    # The min ratio penalizes items that are weak on any constraint
    
    # Normalize to make scores comparable
    geo_mean_norm = geo_mean / (np.max(geo_mean) + 1e-9)
    min_ratio_norm = min_ratio / (np.max(min_ratio) + 1e-9)
    
    # Weighted combination: emphasize the min ratio more to avoid items
    # that are very weak on one constraint
    score = 0.6 * geo_mean_norm + 0.4 * min_ratio_norm
    
    # Also consider the total prize as a tiebreaker
    total_prize = np.sum(prizes, axis=0)  # shape (100,)
    total_prize_norm = total_prize / (np.max(total_prize) + 1e-9)
    
    # Final score: combine efficiency with total prize
    final_score = 0.7 * score + 0.3 * total_prize_norm
    
    # Ensure all values are positive and finite
    final_score = np.where(np.isfinite(final_score) & (final_score > 0), final_score, 1e-9)
    
    return final_score
# EVOLVE-BLOCK-END

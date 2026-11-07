"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s limit
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # Maximum weight across constraints (bottleneck constraint)
    max_weight = weights.max(axis=0)
    
    # Average weight across constraints (overall resource usage)
    avg_weight = weights.mean(axis=0)
    
    # Heuristic: reward high total prize, penalize bottleneck and overall weight
    # Multiplicative penalty: items heavy in any constraint are strongly discouraged
    h = total_prize / ((1.0 + max_weight) * (1.0 + avg_weight) + 1e-9)
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    # Check if we're still within time budget
    if time.time() > deadline:
        return h
    
    # Structural change: Use a different design family
    # Instead of simple ratio-based heuristic, use a normalized score
    # that combines prize density with constraint tightness awareness
    
    # Per-constraint efficiency: prize / weight for each constraint and item
    eff = prizes / (weights + 1e-9)
    
    # Minimum efficiency across constraints (bottleneck efficiency)
    min_eff = eff.min(axis=0)
    
    # Maximum efficiency across constraints (best case efficiency)
    max_eff = eff.max(axis=0)
    
    # Combine: use geometric mean of min and max efficiency
    # This balances between worst-case and best-case constraint performance
    geo_eff = np.sqrt(min_eff * max_eff + 1e-18)
    
    # Normalize total prize to [0, 1] range
    prize_min, prize_max = total_prize.min(), total_prize.max()
    if prize_max > prize_min:
        norm_prize = (total_prize - prize_min) / (prize_max - prize_min)
    else:
        norm_prize = np.ones_like(total_prize)
    
    # Normalize efficiency to [0, 1] range
    eff_min, eff_max = geo_eff.min(), geo_eff.max()
    if eff_max > eff_min:
        norm_eff = (geo_eff - eff_min) / (eff_max - eff_min)
    else:
        norm_eff = np.ones_like(geo_eff)
    
    # New heuristic: weighted combination of normalized prize and efficiency
    # Give more weight to efficiency since it captures constraint interactions
    h_new = 0.3 * norm_prize + 0.7 * norm_eff
    
    # Ensure all values are positive and finite
    h_new = np.where(np.isfinite(h_new) & (h_new > 0), h_new, 1e-9)
    
    # Check time again
    if time.time() > deadline:
        return h
    
    # Return the new heuristic
    return h_new
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of the
    multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) with prize values in [0, 1]
        weights: array of shape (5, 100) with weight values in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Ensure correct shapes and convert to float
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # prizes is (5, 100), weights is (5, 100)
    # We want one score per item (100,)
    
    # Total prize across all constraints for each item
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Identify the bottleneck constraint: max weight per item across constraints
    max_weight = np.max(weights, axis=0)  # shape (100,)
    
    # Also compute per-constraint efficiency for a more nuanced view
    eps = 1e-9
    
    # Per-constraint density: prize_c / (weight_c + eps)
    efficiency = prizes / (weights + eps)  # shape (5, 100)
    
    # Minimum efficiency across constraints (worst-case ratio)
    min_efficiency = efficiency.min(axis=0)  # shape (100,)
    
    # Total density: total_prize / total_weight
    total_weight = weights.sum(axis=0)
    total_density = total_prize / (total_weight + eps)  # shape (100,)
    
    # Strategy: Combine the bottleneck-based heuristic with per-constraint balance
    # The max_weight based ratio captures the binding constraint
    # The min_efficiency captures the worst per-constraint ratio
    # The total_density captures overall value per weight
    
    # Use a geometric mean approach in log-space for numerical stability
    # Weight the bottleneck ratio most heavily, then balance, then overall density
    
    log_bottleneck = np.log(total_prize / (max_weight + eps) + eps)
    log_min_eff = np.log(min_efficiency + eps)
    log_total_den = np.log(total_density + eps)
    
    # Combine: emphasize bottleneck constraint most, then balance, then overall
    # This ensures items that are good in the tightest constraint are preferred,
    # but also rewards items that are balanced across all constraints
    combined_log = 0.5 * log_bottleneck + 0.3 * log_min_eff + 0.2 * log_total_den
    
    heuristic = np.exp(combined_log)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

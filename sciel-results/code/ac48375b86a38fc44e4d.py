"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Uses a combination of strategies:
    1. Per-constraint prize-to-weight ratios
    2. Geometric mean across constraints (bottleneck-aware)
    3. Normalized prize contribution
    4. Weight penalty for items that are heavy in any constraint
    """
    eps = 1e-9
    
    # Per-constraint prize/weight ratio for each item
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Geometric mean across constraints (bottleneck-aware)
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_mean = np.exp(log_geo_mean)
    
    # Arithmetic mean of ratios
    arith_mean = ratios.mean(axis=0)
    
    # Minimum ratio (bottleneck constraint)
    min_ratio = ratios.min(axis=0)
    
    # Normalize prizes: total prize per item relative to max
    total_prizes = prizes.sum(axis=0)
    prize_norm = total_prizes / (total_prizes.max() + eps)
    
    # Weight penalty: items with high weight in any constraint get penalized
    # Use the max weight across constraints as a penalty factor
    max_weights = weights.max(axis=0)
    weight_penalty = 1.0 / (1.0 + max_weights)
    
    # Combine: geometric mean is primary, boosted by prize norm and weight penalty
    # The geometric mean already captures cross-constraint efficiency
    # We blend with min_ratio to emphasize the bottleneck
    combined = 0.5 * geo_mean + 0.3 * min_ratio + 0.2 * arith_mean
    
    # Scale by prize normalization and weight penalty
    scores = combined * prize_norm * weight_penalty
    
    # Ensure all values are positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

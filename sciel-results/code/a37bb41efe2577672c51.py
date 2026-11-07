"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy: Constraint-specific scarcity premium
    1. Identify the most binding constraint per item (largest weight fraction)
    2. Scale prize-to-weight ratio by inverse of remaining capacity for that constraint
    3. Blend with geometric mean (smooth min) for robustness
    """
    eps = 1e-9
    capacity = 25.0
    
    # Per-constraint prize/weight ratio for each item
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Geometric mean across constraints (bottleneck-aware, smooth min)
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_mean = np.exp(log_geo_mean)
    
    # Identify most binding constraint per item
    # The constraint where weight is largest fraction of total weight
    total_weight_per_item = weights.sum(axis=0)  # (100,)
    # Fraction of total weight in each constraint: (5, 100)
    weight_fractions = weights / (total_weight_per_item + eps)
    
    # Most binding constraint index per item: (100,)
    most_binding_idx = np.argmax(weight_fractions, axis=0)  # (100,)
    
    # Get the weight in the most binding constraint for each item
    # Use advanced indexing: weights[most_binding_idx, np.arange(100)]
    item_indices = np.arange(weights.shape[1])
    binding_weights = weights[most_binding_idx, item_indices]  # (100,)
    
    # Remaining capacity fraction for the most binding constraint
    # remaining = capacity - binding_weight, fraction = remaining/capacity
    remaining_capacity = capacity - binding_weights
    remaining_fraction = remaining_capacity / capacity
    
    # Scarcity premium: inverse of remaining fraction (higher when constraint is tight)
    # Clip to avoid division by zero or negative values
    remaining_fraction = np.maximum(remaining_fraction, eps)
    scarcity_premium = 1.0 / remaining_fraction
    
    # Prize-to-weight ratio in the most binding constraint
    binding_ratios = ratios[most_binding_idx, item_indices]  # (100,)
    
    # Scarcity-adjusted ratio: high value relative to most limiting resource
    scarcity_adjusted = binding_ratios * scarcity_premium
    
    # Blend scarcity-adjusted ratio with geometric mean
    # Normalize both to mean 1 for fair blending
    scarcity_mean = scarcity_adjusted.mean()
    geo_mean_val = geo_mean.mean()
    
    scarcity_norm = scarcity_adjusted / (scarcity_mean + eps)
    geo_norm = geo_mean / (geo_mean_val + eps)
    
    # Blend: emphasize scarcity premium but keep geo mean for stability
    # 60% scarcity-adjusted, 40% geometric mean
    scores = 0.6 * scarcity_norm + 0.4 * geo_norm
    
    # Ensure all values are positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

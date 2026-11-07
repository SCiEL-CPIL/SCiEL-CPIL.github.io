"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy: Combines aggregate efficiency, bottleneck (geometric mean) efficiency,
    and constraint-specific scarcity premium.
    
    1. Aggregate Efficiency: Total prize / total weight.
    2. Bottleneck Efficiency: Geometric mean of per-constraint prize/weight ratios.
    3. Scarcity Premium: For each item, identify the most binding constraint 
       (largest weight fraction) and scale its ratio by inverse remaining capacity.
    
    All metrics are normalized to mean 1 and blended with weights tuned to 
    balance overall value, balance across constraints, and scarcity awareness.
    """
    eps = 1e-9
    capacity = 25.0
    
    # Ensure positive values to avoid log/division issues
    w = np.clip(weights, 1e-12, None)
    p = np.clip(prizes, 1e-12, None)
    
    # 1. Aggregate Efficiency: sum of prizes / sum of weights
    total_weights = w.sum(axis=0)  # (100,)
    total_prizes = p.sum(axis=0)  # (100,)
    sum_ratio = total_prizes / (total_weights + eps)  # (100,)
    
    # 2. Bottleneck Efficiency: Geometric mean of per-constraint ratios
    ratios = p / (w + eps)  # (5, 100)
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_ratio = np.exp(log_geo_mean)  # (100,)
    
    # 3. Constraint-specific Scarcity Premium
    # Identify most binding constraint per item (largest weight fraction)
    weight_fractions = w / (total_weights + eps)  # (5, 100)
    most_binding_idx = np.argmax(weight_fractions, axis=0)  # (100,)
    
    item_indices = np.arange(w.shape[1])
    binding_weights = w[most_binding_idx, item_indices]  # (100,)
    binding_ratios = ratios[most_binding_idx, item_indices]  # (100,)
    
    # Remaining capacity fraction for the most binding constraint
    remaining_capacity = capacity - binding_weights
    remaining_fraction = np.maximum(remaining_capacity / capacity, eps)
    
    # Scarcity premium: high when remaining capacity is low
    scarcity_premium = 1.0 / remaining_fraction
    
    # Scarcity-adjusted ratio
    scarcity_adjusted = binding_ratios * scarcity_premium  # (100,)
    
    # Normalize all three metrics to mean 1
    def normalize_mean(arr):
        mean_val = arr.mean()
        if mean_val > 0:
            return arr / mean_val
        return np.ones_like(arr)
    
    sum_ratio_norm = normalize_mean(sum_ratio)
    geo_ratio_norm = normalize_mean(geo_ratio)
    scarcity_norm = normalize_mean(scarcity_adjusted)
    
    # Blend the three metrics
    # Weights: 40% aggregate, 35% bottleneck, 25% scarcity
    # This balances overall value, cross-constraint balance, and scarcity awareness
    scores = 0.4 * sum_ratio_norm + 0.35 * geo_ratio_norm + 0.25 * scarcity_norm
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

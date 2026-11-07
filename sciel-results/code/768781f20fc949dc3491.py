"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of the
    multiple-constraint knapsack.
    
    Strategy: constraint-specific scarcity premium.
    For each item, identify the most binding constraint (where its weight is
    the largest fraction of the item's total weight) and scale its prize-to-weight
    ratio by the inverse of the remaining capacity fraction for that constraint.
    
    prizes, weights: shape (5, 100), uniform in [0, 1]
    Each constraint has capacity 25.
    
    Returns: array of 100 scores, one per item (larger = more attractive)
    """
    start_time = time.time()
    deadline = start_time + 320.0  # safety margin before 330s kill
    
    C = 25.0  # capacity per constraint
    n_items = weights.shape[1]  # 100
    
    # Total weight per item across all constraints
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # Prize-to-total-weight ratio (base value density)
    # Avoid division by zero
    base_ratio = prizes.sum(axis=0) / (total_weight + 1e-9)
    
    # For each item, find the most binding constraint:
    # the constraint where weight[i] is the largest fraction of total_weight[i]
    # fraction[i, j] = weights[j, i] / total_weight[i]
    # The most binding constraint for item i is argmax_j fraction[i, j]
    
    # Compute fractions: shape (5, 100)
    safe_total = total_weight + 1e-9
    fractions = weights / safe_total  # shape (5, 100)
    
    # For each item, the most binding constraint index
    most_binding = np.argmax(fractions, axis=0)  # shape (100,)
    
    # The fraction of capacity consumed by the most binding constraint for each item
    # For item i, constraint j = most_binding[i], the weight is weights[j, i]
    # The capacity used fraction is weights[j, i] / C
    # The remaining capacity fraction is 1 - weights[j, i] / C
    # Scarcity premium = 1 / (remaining capacity fraction) = C / (C - weights[j, i])
    
    # Get the weight of the most binding constraint for each item
    # most_binding[i] is the constraint index for item i
    # weights[most_binding, np.arange(n_items)] gives the weight in that constraint
    binding_weights = weights[most_binding, np.arange(n_items)]  # shape (100,)
    
    # Remaining capacity fraction for the binding constraint
    remaining_fraction = 1.0 - binding_weights / C  # shape (100,)
    # Clip to avoid division by zero or negative values
    remaining_fraction = np.maximum(remaining_fraction, 1e-6)
    
    # Scarcity premium: higher when the item consumes more of its most binding constraint
    scarcity_premium = 1.0 / remaining_fraction  # shape (100,)
    
    # Combine: base ratio scaled by scarcity premium
    # This prioritizes items that offer high value relative to their most limiting resource
    scores = base_ratio * scarcity_premium
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    # Check time budget
    if time.time() > deadline:
        return scores
    
    # Normalize scores to be on a reasonable scale
    # The heuristic is used as pheromone * heuristic in probability,
    # so relative magnitudes matter more than absolute
    # But let's ensure they're well-scaled
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score  # normalize to [0, 1] range roughly
    
    # Ensure minimum positive value
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END

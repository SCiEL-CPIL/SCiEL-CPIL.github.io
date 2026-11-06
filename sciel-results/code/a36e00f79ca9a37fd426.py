"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    # Filter out any non-positive weights or values (safety)
    mask = (weights > 0) & (values > 0)
    if not np.any(mask):
        return -1
    
    weights = weights[mask]
    values = values[mask]
    
    # Modified density score: base density boosted by capacity utilization
    # Items that use a larger fraction of remaining capacity get a bonus
    density = values / weights
    utilization = np.minimum(1.0, remaining_capacity / weights)
    score = density * (1.0 + 0.5 * utilization)
    
    best_idx = int(np.argmax(score))
    
    # Map back to original index
    original_indices = np.where(mask)[0]
    return int(original_indices[best_idx])
# EVOLVE-BLOCK-END

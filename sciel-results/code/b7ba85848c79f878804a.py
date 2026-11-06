"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit
_SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline guard
    if time.time() > _start_time + _DEADLINE - _SAFETY_MARGIN:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Only consider items that fit
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    # Composite score: value density with capacity-utilization discount
    # score = (value / weight) * (capacity / (capacity + weight))
    # This favors high density but penalizes items that waste a lot of remaining capacity
    
    w = remaining_weights[feasible_mask]
    v = remaining_values[feasible_mask]
    
    # Density
    density = v / w
    
    # Capacity utilization factor: 1.0 when weight << capacity, ~0.5 when weight ~= capacity
    util_factor = remaining_capacity / (remaining_capacity + w)
    
    score = density * util_factor
    
    # Add small bonus for absolute value to break ties favoring high-value items
    # Scale bonus relative to typical value to keep it as a tiebreaker
    max_v = np.max(v) if len(v) > 0 else 1.0
    score += 0.01 * (v / max_v)
    
    # Pick the best feasible item
    best_idx_in_feasible = int(np.argmax(score))
    
    # Map back to original index
    feasible_indices = np.where(feasible_mask)[0]
    return int(feasible_indices[best_idx_in_feasible])
# EVOLVE-BLOCK-END

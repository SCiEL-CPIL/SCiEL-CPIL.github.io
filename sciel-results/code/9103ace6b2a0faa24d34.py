import numpy as np
import time

_start_time = None

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 Knapsack:
    1. Calculate value density for each feasible item.
    2. Use a composite score that balances density and weight.
    3. Prioritize items that maximize value while preserving capacity.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return -1 if we're getting close to the limit
    if time.time() - _start_time > 145:
        return -1

    if len(remaining_weights) == 0:
        return -1

    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = remaining_values / remaining_weights
    
    # Handle zero weights
    densities = np.where(densities == np.inf, 1e9, densities)
    
    # Simple density-based selection with tie-breaking by weight
    # This is more robust than the complex tiered approach
    # We pick the item with highest density, breaking ties by lower weight
    
    # Create a composite score: density * (1 / (1 + weight/capacity))
    # This slightly penalizes heavier items to preserve capacity
    # But primarily follows density
    
    # Alternative: Just use density, but break ties by weight
    # This is often near-optimal for knapsack heuristics
    
    # Let's try a weighted combination
    # Score = density * (1 - 0.1 * weight / capacity)
    # This gives 10% preference to lighter items among similar densities
    
    if remaining_capacity <= 0:
        return -1
        
    # Normalize weights relative to capacity
    weight_penalty = 0.1 * remaining_weights / remaining_capacity
    weight_penalty = np.clip(weight_penalty, 0, 1)
    
    composite_score = densities * (1 - weight_penalty)
    
    # Find the item with highest composite score
    best_idx = np.argmax(composite_score)
    
    return int(best_idx)

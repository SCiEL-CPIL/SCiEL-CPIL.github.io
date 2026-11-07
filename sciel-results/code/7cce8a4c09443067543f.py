import numpy as np
import time

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Capacity per constraint is 25
    capacity = 25.0
    num_constraints, num_items = weights.shape
    
    # For each item, compute the fraction of capacity consumed in each constraint
    weight_fractions = weights / capacity  # shape (num_constraints, num_items)
    
    # Total prize for each item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)  # shape (num_items,)
    
    # Compute a more nuanced density metric:
    # For each constraint, compute the "cost" of an item in that constraint
    # and weight it by how tight that constraint is likely to be
    
    # One approach: use the sum of (prize_in_constraint / weight_in_constraint) 
    # but that's tricky. Instead, let's try a different heuristic.
    
    # Alternative: For each item, compute the minimum "efficiency" across constraints
    # where efficiency = prize / weight for each constraint
    
    # Actually, let's try: total_prize / (sum of weights across all constraints)
    # This gives a more global density measure
    
    sum_weights = np.sum(weights, axis=0)  # shape (num_items,)
    eps = 1e-9
    
    # Try multiple heuristics and pick the best one based on a simple criterion
    # Heuristic 1: total_prize / max_weight_fraction
    max_fraction = np.max(weight_fractions, axis=0)
    h1 = total_prize / (max_fraction + eps)
    
    # Heuristic 2: total_prize / sum_weights
    h2 = total_prize / (sum_weights + eps)
    
    # Heuristic 3: Geometric mean of per-constraint efficiencies
    # For each constraint c and item i, efficiency = prizes[c,i] / weights[c,i]
    # But this can be unstable. Let's use a smoothed version.
    
    # Actually, let's try: for each item, compute the minimum ratio 
    # (prizes[c,i] / weights[c,i]) across constraints where weights > 0
    # This captures the bottleneck constraint's efficiency
    
    # To avoid division by zero, add a small epsilon
    safe_weights = np.where(weights > eps, weights, 1.0)
    ratios = prizes / safe_weights  # shape (num_constraints, num_items)
    
    # For each item, find the minimum ratio across constraints
    # But only consider constraints where the item actually uses capacity
    min_ratios = np.min(ratios, axis=0)  # shape (num_items,)
    
    # Combine: use total_prize * min_ratio / (sum_weights + eps)
    h3 = total_prize * min_ratios / (sum_weights + eps)
    
    # Ensure all heuristics are positive and finite
    for h in [h1, h2, h3]:
        h[np.where(~np.isfinite(h))] = eps
        h[h < eps] = eps
    
    # Choose the heuristic that gives the best "spread" 
    # or just use a weighted combination
    
    # Let's use a weighted combination with emphasis on h2 (global density)
    # and h1 (bottleneck-aware)
    heuristic = 0.5 * h1 + 0.5 * h2
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic

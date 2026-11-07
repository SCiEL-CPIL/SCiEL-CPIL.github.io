"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s margin
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # prizes: (5, 100), weights: (5, 100)
    n_items = prizes.shape[1]
    
    eps = 1e-9
    
    # Total prize per item: sum across all 5 prize dimensions
    total_prize = np.sum(prizes, axis=0)  # shape (100,)
    
    # Maximum weight per item: the bottleneck constraint
    max_weight = np.max(weights, axis=0)  # shape (100,)
    
    # Sum of weights per item: total resource consumption
    sum_weight = np.sum(weights, axis=0)  # shape (100,)
    
    # Approach: prize efficiency relative to worst-case constraint
    # Use max_weight as the binding constraint
    safe_max_weight = np.maximum(max_weight, eps)
    safe_sum_weight = np.maximum(sum_weight, eps)
    
    # Multiple efficiency measures
    eff_max = total_prize / safe_max_weight  # prize per worst-case unit
    eff_sum = total_prize / safe_sum_weight  # overall efficiency
    
    # Combine: weight the worst-case efficiency more heavily since it's the binding constraint
    # Also consider that items with low absolute weight are easier to fit
    # Normalize each to [0,1] range
    eff_max_norm = (eff_max - np.min(eff_max)) / (np.max(eff_max) - np.min(eff_max) + eps)
    eff_sum_norm = (eff_sum - np.min(eff_sum)) / (np.max(eff_sum) - np.min(eff_sum) + eps)
    
    # Also consider absolute prize (high prize items are good)
    prize_norm = (total_prize - np.min(total_prize)) / (np.max(total_prize) - np.min(total_prize) + eps)
    
    # Combine with weights: worst-case efficiency is most important
    heuristic_scores = 0.5 * eff_max_norm + 0.3 * eff_sum_norm + 0.2 * prize_norm
    
    # Ensure strictly positive
    heuristic_scores = np.maximum(heuristic_scores, eps)
    
    # Handle non-finite values
    heuristic_scores = np.where(np.isfinite(heuristic_scores), heuristic_scores, eps)
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    return heuristic_scores
# EVOLVE-BLOCK-END

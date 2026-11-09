"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins is the array of remaining capacities for each bin.
    # The item goes to the highest-scoring bin among those with enough capacity.
    
    remaining = bins
    capacity = 100.0
    
    # Only consider bins that can fit the item
    mask = remaining >= item
    if not np.any(mask):
        # No bin can fit the item; return all -inf
        return np.full_like(remaining, -np.inf)
    
    # Get the residuals for bins that can fit the item
    residual = remaining[mask] - item
    
    # Adaptive Strategy:
    # Estimate the "phase" of the packing.
    # If the average remaining capacity of *all* bins is high, we are early in the process
    # or items are small, so we should preserve space (Worst-Fit).
    # If the average remaining capacity is low, we are late or items are large, 
    # so we should consolidate (Best-Fit).
    
    # Use the mean of ALL bins to determine the global state
    mean_remaining = np.mean(remaining)
    
    # Threshold: If mean remaining is > 50, we have lots of space, use Worst-Fit.
    # If mean remaining is <= 50, we are getting tight, use Best-Fit.
    # We can smooth this transition.
    
    # Score calculation
    # Worst-Fit: score = residual (maximize)
    # Best-Fit: score = -residual (minimize)
    
    # Let's use a weighted blend based on mean_remaining
    # weight_worst = mean_remaining / capacity
    # weight_best = 1 - weight_worst
    
    # score = weight_worst * residual - weight_best * residual
    # score = (weight_worst - weight_best) * residual
    # weight_worst - weight_best = 2 * (mean_remaining / capacity) - 1
    
    # If mean_remaining = 100, factor = 1 -> score = residual (Worst-Fit)
    # If mean_remaining = 0, factor = -1 -> score = -residual (Best-Fit)
    # If mean_remaining = 50, factor = 0 -> score = 0 (Tie, breaks arbitrarily or by index)
    
    factor = 2.0 * (mean_remaining / capacity) - 1.0
    
    score = factor * residual
    
    # Map back to full array
    full_score = np.full_like(remaining, -np.inf)
    full_score[mask] = score
    
    return full_score
# EVOLVE-BLOCK-END

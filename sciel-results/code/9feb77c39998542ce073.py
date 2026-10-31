import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=float)
    
    # Mask invalid bins (remaining capacity < item)
    valid = bins >= item
    
    # Compute global statistics of remaining capacities
    valid_bins = bins[valid]
    if len(valid_bins) > 0:
        mean_remaining = np.mean(valid_bins)
        var_remaining = np.var(valid_bins)
    else:
        mean_remaining = 0.0
        var_remaining = 1.0
    
    # For each valid bin, compute a score that combines:
    # 1. Best-fit component: prefer bins where remaining capacity is close to item size
    # 2. Global alignment component: prefer bins whose remaining capacity aligns with the global mean
    #    and penalize bins that deviate significantly from the global distribution
    
    # Best-fit score: negative of how much extra space would be left
    best_fit_score = -(bins - item)
    
    # Global alignment score: prefer bins close to the mean remaining capacity
    # This helps consolidate loads and reduce fragmentation
    alignment_score = -np.abs(bins - mean_remaining)
    
    # Variance penalty: bins with remaining capacity far from the mean are more "fragmenting"
    # Use a soft penalty based on how many standard deviations away the bin is
    if var_remaining > 1e-10:
        std_dev = np.sqrt(var_remaining)
        z_score = (bins - mean_remaining) / std_dev
        variance_penalty = -0.1 * np.abs(z_score)
    else:
        variance_penalty = np.zeros_like(bins)
    
    # Combine scores with weights
    # Weight best-fit more heavily, but include global alignment
    combined_score = 0.7 * best_fit_score + 0.2 * alignment_score + 0.1 * variance_penalty
    
    # For invalid bins, set score to -inf
    scores = np.where(valid, combined_score, -np.inf)
    
    return scores

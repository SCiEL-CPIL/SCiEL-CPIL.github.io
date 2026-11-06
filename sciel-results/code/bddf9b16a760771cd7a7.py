import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Next-Item Awareness heuristic:
    # Estimate expected next item size from current bin occupancy distribution,
    # then score bins by how well their post-placement remaining space matches
    # this predicted size (minimizing waste probability).
    
    remaining = bins - item
    
    # Identify valid bins (enough capacity for current item)
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        # No valid bins; return all -inf (shouldn't happen in normal operation)
        return np.full_like(remaining, -np.inf, dtype=float)
    
    # Estimate expected next item size from occupied bins
    # Use the mean of (capacity - bin) for bins that have some occupancy
    # This gives a rough estimate of typical item sizes seen so far
    capacity = 100.0
    occupancy = capacity - bins
    occupied_mask = occupancy > 1e-10  # bins with some items
    if np.any(occupied_mask):
        # Estimate next item size as mean occupancy of occupied bins
        # Weibull items tend to be smaller, so use a trimmed mean to reduce outlier impact
        occ_vals = occupancy[occupied_mask]
        if len(occ_vals) >= 10:
            # Use 25th-75th percentile range for robustness
            q25, q75 = np.percentile(occ_vals, [25, 75])
            filtered = occ_vals[(occ_vals >= q25) & (occ_vals <= q75)]
            if len(filtered) > 0:
                next_item_est = np.mean(filtered)
            else:
                next_item_est = np.mean(occ_vals)
        else:
            next_item_est = np.mean(occ_vals)
    else:
        # No items placed yet; assume moderate item size
        next_item_est = 25.0
    
    # For valid bins, compute post-placement remaining space
    post_remaining = remaining[valid_mask]
    
    # Score: prefer bins where post_remaining is close to next_item_est
    # Use negative absolute difference as score (higher = better)
    # But also penalize if post_remaining is much larger than needed (waste)
    # and if it's negative (invalid, already masked)
    
    # Smooth scoring: -|post_remaining - next_item_est| with a penalty for large waste
    diff = np.abs(post_remaining - next_item_est)
    # Penalize excessive remaining space (more than 2x expected next item)
    waste_penalty = np.maximum(0, post_remaining - 2 * next_item_est) * 0.1
    
    valid_scores = -diff - waste_penalty
    
    # Build full score array
    scores = np.full_like(remaining, -np.inf, dtype=float)
    scores[valid_mask] = valid_scores
    
    return scores

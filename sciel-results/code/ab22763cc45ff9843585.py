import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    Uses a hybrid strategy with:
    - Best Fit for small items, Worst Fit for large items
    - Penalty for creating very small remainders (< 5) in Best Fit
    - Global variance-based risk assessment to reduce fragmentation
    """
    n_bins = len(bins)
    # Initialize all scores to a very low value (invalid for selection)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    # If no valid bins, return the low scores
    if not np.any(valid_mask):
        return scores
        
    # Extract remaining capacities of valid bins
    valid_remaining = bins[valid_mask]
    
    # Capacity is 100. Use a smooth transition around 50.
    # Smooth sigmoid-like blend: weight for Best Fit vs Worst Fit
    # When item is small, favor Best Fit (minimize remaining after placement)
    # When item is large, favor Worst Fit (maximize remaining)
    
    # Compute global statistics for variance-based risk assessment
    if n_bins > 1:
        mean_remaining = np.mean(bins)
        std_remaining = np.std(bins)
        if std_remaining < 1e-9:
            # All bins have similar remaining capacity
            variance_risk = np.zeros(n_bins)
        else:
            # Smooth Gaussian-like risk: penalize bins far from the mean
            # This encourages placing items in bins that are "typical"
            # reducing the spread of remaining capacities (fragmentation)
            z_scores = (bins - mean_remaining) / std_remaining
            variance_risk = -np.exp(-0.5 * z_scores ** 2)
    else:
        variance_risk = np.zeros(n_bins)
    
    # Blend weight: smoothly transitions from Best Fit (small items) to Worst Fit (large items)
    # Use a logistic function centered at 50 with some width
    center = 50.0
    width = 10.0
    # When item << 50, blend ~ 1 (Best Fit dominant)
    # When item >> 50, blend ~ 0 (Worst Fit dominant)
    blend = 1.0 / (1.0 + np.exp((item - center) / width))
    
    # For Best Fit: score = -(remaining after placement) = -(bins - item) = item - bins
    # For Worst Fit: score = remaining after placement = bins - item
    
    # Best Fit component: prefer bins with least remaining space after item placement
    best_fit_score = item - valid_remaining  # negative of (bins - item)
    
    # Worst Fit component: prefer bins with most remaining space after item placement  
    worst_fit_score = valid_remaining - item  # positive (bins - item)
    
    # Blended base score
    base_score = blend * best_fit_score + (1.0 - blend) * worst_fit_score
    
    # Penalty for very small remainders in Best Fit mode
    # If the remaining space after placing the item is very small, it's likely unusable
    # Apply a smooth penalty that kicks in when remaining < 5
    remaining_after = valid_remaining - item
    # Smooth penalty: 0 when remaining_after >= 5, grows as remaining_after approaches 0
    # Use a quadratic penalty for remaining_after < 5
    small_remainder_penalty = np.where(
        remaining_after < 5.0,
        -0.5 * (5.0 - remaining_after) ** 2,  # Max penalty of -12.5 at remaining_after=0
        0.0
    )
    # Scale the penalty by how much we're using Best Fit
    penalty_weighted = blend * small_remainder_penalty
    
    # Variance risk term: favor bins close to the mean remaining capacity
    # Weight this more for small items (consolidation) and less for large items
    variance_weight = 0.3 * blend + 0.1 * (1.0 - blend)
    variance_term = variance_weight * variance_risk[valid_mask]
    
    # Final score
    scores[valid_mask] = base_score + penalty_weighted + variance_term
    
    return scores

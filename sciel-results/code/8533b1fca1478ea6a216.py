import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    # Strategy: Structural Centrality / Variance Minimization
    # Score is negative squared difference between the bin's new remainder
    # and the mean remainder of the other open bins.
    
    # Precompute total sum of remainders for efficiency
    total_sum = np.sum(remaining)
    n_bins = len(remaining)
    
    # Avoid division by zero if only one bin is open (though valid check ensures at least 1)
    # If n_bins == 1, mean_others is undefined, but variance is 0. 
    # We fall back to Best Fit or 0 score? 
    # If n_bins == 1, there are no "other" bins. 
    # Let's treat mean_others as 0 or just use Best Fit for n=1.
    # Actually, if n=1, we must pick it. Score can be 0.
    
    if n_bins > 1:
        # mean_others[i] = (total_sum - remaining[i]) / (n_bins - 1)
        # We only need this for valid bins, but vectorized is easier for all.
        # Note: remaining is the array of *current* remainders before placement.
        # Wait, `remaining` variable in the code above is `bins - item`.
        # This is the *new* remainder.
        # The "other" bins have their *current* remainders, which are `bins[j]`.
        # So we need the mean of `bins[j]` for j != i.
        
        # Let's re-read carefully.
        # "sum of squared differences between its resulting remainder and the mean remainder of all other open bins"
        # Resulting remainder for bin i: `bins[i] - item`
        # Other open bins: bins j where j != i. Their remainders are `bins[j]`.
        # So we need `mean(bins[j] for j != i)`.
        
        total_sum_original = np.sum(bins)
        mean_others = (total_sum_original - bins) / (n_bins - 1)
        
        # For valid bins:
        # new_rem = remaining[valid]  (which is bins[valid] - item)
        # target = mean_others[valid]
        
        valid_indices = np.where(valid)[0]
        
        new_rems = remaining[valid_indices]
        targets = mean_others[valid_indices]
        
        # Score = - (new_rem - target)^2
        scores[valid_indices] = -np.square(new_rems - targets)
        
        # Tie-breaking: Prefer lower index?
        # Add tiny penalty
        scores[valid_indices] -= np.arange(len(bins))[valid_indices] * 1e-9

    else:
        # Only one bin, just pick it
        scores[0] = 0.0

    return scores

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a probabilistic risk model.
    
    The approach models bin selection as a risk assessment based on global variance:
    - Compute the global mean of remaining capacities across all open bins.
    - For each candidate bin, calculate what the new global variance would be
      if this bin were selected (i.e., its remaining capacity decreases by item).
    - Score bins by the negative of this hypothetical new variance: bins that
      reduce overall variance are preferred, as they keep the load distribution
      tight and reduce fragmentation.
    
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Get the remaining capacities of valid bins
    valid_remaining = remaining[valid_mask]
    
    # If no valid bins, return all negative infinity (shouldn't happen in practice)
    if len(valid_remaining) == 0:
        return np.full(len(bins), -np.inf)
    
    # Compute global statistics of remaining capacities across ALL bins
    all_remaining = bins
    n = len(all_remaining)
    global_mean = np.mean(all_remaining)
    
    # For each valid bin i, if we place the item in bin i:
    # The new remaining capacity of bin i becomes: bins[i] - item
    # The new global mean becomes: (sum(bins) - item) / n = global_mean - item/n
    # The new variance can be computed as:
    # new_var = (1/n) * sum((bins[j] - new_mean)^2) for all j
    #        = (1/n) * [sum_{j != i} (bins[j] - new_mean)^2 + (bins[i] - item - new_mean)^2]
    
    # We can compute this efficiently:
    # Let S = sum(bins), Q = sum(bins^2)
    # new_mean = (S - item) / n
    # new_Q = Q - bins[i]^2 + (bins[i] - item)^2
    # new_var = new_Q / n - new_mean^2
    
    S = np.sum(all_remaining)
    Q = np.sum(all_remaining ** 2)
    
    # For each valid bin, compute the hypothetical new variance
    # new_Q_i = Q - bins[i]^2 + (bins[i] - item)^2
    # new_mean_i = (S - item) / n (same for all i)
    # new_var_i = new_Q_i / n - new_mean_i^2
    
    # Since new_mean_i is the same for all i, minimizing new_var_i is equivalent
    # to minimizing new_Q_i, which is equivalent to minimizing:
    # Q - bins[i]^2 + (bins[i] - item)^2
    # = Q + bins[i]^2 - 2*item*bins[i] + item^2 - bins[i]^2
    # = Q + item^2 - 2*item*bins[i]
    
    # So minimizing new_Q_i is equivalent to minimizing -2*item*bins[i],
    # which means maximizing bins[i] (the original remaining capacity).
    # Wait, that doesn't seem right. Let me re-derive.
    
    # Actually, let me just compute the scores directly:
    new_mean = (S - item) / n
    
    # For each valid bin index, compute the new variance
    valid_indices = np.where(valid_mask)[0]
    
    # bins[valid_indices] are the original remaining capacities
    # After placement, they become bins[valid_indices] - item
    
    # new_Q_i = Q - bins[i]^2 + (bins[i] - item)^2
    valid_bins_orig = all_remaining[valid_mask]
    
    new_Q = Q - valid_bins_orig ** 2 + (valid_bins_orig - item) ** 2
    new_var = new_Q / n - new_mean ** 2
    
    # Score is negative new variance (we want to minimize variance)
    combined_score = -new_var
    
    # Create the full score array
    scores = np.full(len(bins), -np.inf)
    scores[valid_mask] = combined_score
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a Next-Item Awareness heuristic.
    
    The evaluator picks the bin with the highest score among bins with enough remaining capacity.
    We want to minimize the number of bins used (excess over L1 lower bound).
    
    Strategy:
    - Estimate the expected next item size from the current bin state
    - Score bins based on how well their remaining space matches this predicted next item
    - Use a smooth Gaussian kernel for the match score
    - Combine with best-fit (tight fit) preference
    - Add a small quadratic penalty on remaining space to favor tight fits
    """
    deadline = time.time() + 0.5  # 0.5 second budget with safety margin
    
    remaining = bins - item
    
    # Filter: only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # If no valid bins, return scores that will result in no selection
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    # Estimate the expected next item size from the current bin occupancy
    # Use the median of the remaining spaces as a proxy for the typical item size
    # Blend with the current item size for stability
    median_remaining = np.median(valid_remaining)
    
    # The expected next item size is a blend of the current item and typical remaining space
    # This assumes items are drawn from a similar distribution
    expected_next_item = 0.6 * median_remaining + 0.4 * item
    
    # Ensure expected_next_item is positive and reasonable
    expected_next_item = max(expected_next_item, 1.0)
    expected_next_item = min(expected_next_item, 100.0)
    
    # Compute the standard deviation of remaining spaces to set the Gaussian width
    std_remaining = np.std(valid_remaining)
    # Use a fraction of the std as the Gaussian width for smooth scoring
    sigma = max(0.25 * std_remaining, 2.0)  # Minimum width to avoid degenerate scores
    
    # Next-item awareness score using Gaussian kernel:
    # Higher score when remaining space closely matches expected next item size
    diff = valid_remaining - expected_next_item
    next_item_score = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Best-fit score: prefer bins with less remaining space (tighter fit)
    # Normalize to [0, 1] range
    max_remaining = np.max(valid_remaining)
    min_remaining = np.min(valid_remaining)
    if max_remaining > min_remaining:
        best_fit_score = 1.0 - (valid_remaining - min_remaining) / (max_remaining - min_remaining)
    else:
        best_fit_score = np.ones(n_valid)
    
    # Combine the two scores with weights
    # Weight 0.7 for best-fit, weight 0.3 for next-item awareness
    combined_score = 0.7 * best_fit_score + 0.3 * next_item_score
    
    # Add a small quadratic penalty on remaining space to further favor tight fits
    capacity_estimate = 100.0
    quad_penalty = 0.01 * (valid_remaining ** 2) / capacity_estimate
    
    final_scores = combined_score - quad_penalty
    
    scores[valid_mask] = final_scores
    
    # Check deadline
    if time.time() > deadline:
        # Return whatever we have
        pass
    
    return scores
# EVOLVE-BLOCK-END

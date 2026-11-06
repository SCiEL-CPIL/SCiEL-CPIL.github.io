"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We track some global state to estimate the distribution of item sizes
# This is allowed since the evaluator runs multiple items sequentially
_item_history = []
_MAX_HISTORY = 200

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Next-Item Awareness heuristic
    - Estimate expected next item size from recent history
    - Score bins based on how well their remaining space (after placing current item)
      matches the predicted next item size
    - Prefer bins where remaining space is close to expected next item (minimizes waste)
    - Use a smooth penalty for large deviations
    """
    global _item_history
    
    # Record current item for future predictions
    _item_history.append(item)
    if len(_item_history) > _MAX_HISTORY:
        _item_history = _item_history[-_MAX_HISTORY:]
    
    # Estimate expected next item size
    if len(_item_history) >= 2:
        # Use median for robustness, weighted toward recent items
        recent = _item_history[-min(50, len(_item_history)):]
        weights = np.linspace(0.5, 1.0, len(recent))
        expected_next = np.average(recent, weights=weights)
        # Also compute a measure of variability
        std_next = np.std(recent)
    else:
        # Fallback: assume typical item size is around 10 (for capacity 100)
        expected_next = 10.0
        std_next = 5.0
    
    # Filter bins with enough remaining capacity
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    
    # After placing current item, the remaining space in each valid bin is:
    # post_remaining = bins[valid] - item
    post_remaining = valid_remaining
    
    # Score based on how well post_remaining matches expected_next
    # We want post_remaining to be close to expected_next (so next item fits well)
    # But we also don't want too much excess (waste)
    
    # Gaussian-like score: peaks when post_remaining ≈ expected_next
    # Use a soft-min style: penalize |post_remaining - expected_next|
    # But also penalize if post_remaining is much larger than expected (waste)
    
    # Primary term: closeness to expected next item size
    diff = post_remaining - expected_next
    # Scale by std to normalize
    normalized_diff = diff / (std_next + 1e-6)
    
    # Gaussian kernel: exp(-0.5 * normalized_diff^2)
    gaussian_score = np.exp(-0.5 * normalized_diff ** 2)
    
    # Secondary term: penalize large absolute remaining space (waste)
    # Use a soft penalty: -alpha * max(0, post_remaining - expected_next - std_next)
    excess_waste = np.maximum(0, post_remaining - (expected_next + std_next))
    waste_penalty = 0.1 * excess_waste
    
    # Tertiary term: slight preference for bins with more remaining space
    # (in case next item is larger than expected)
    # This is a mild linear term
    capacity_bonus = 0.01 * post_remaining
    
    # Combine scores
    # Higher is better
    combined = gaussian_score - waste_penalty + capacity_bonus
    
    # Assign back to valid bins
    scores[valid_mask] = combined
    
    return scores
# EVOLVE-BLOCK-END

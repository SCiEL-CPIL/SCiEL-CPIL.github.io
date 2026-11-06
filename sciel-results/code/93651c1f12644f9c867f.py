"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track recent item sizes for "Next-Item Awareness"
_recent_items = []
_MAX_RECENT = 200

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: "Next-Item Awareness" Best Fit
    - Estimate the expected size of the next item from recent history.
    - Score bins based on how well their remaining space (after placing current item)
      matches the predicted next item size, while still favoring tight packing.
    """
    global _recent_items
    
    # Track recent items for prediction
    _recent_items.append(item)
    if len(_recent_items) > _MAX_RECENT:
        _recent_items = _recent_items[-_MAX_RECENT:]
    
    # Calculate remaining capacity for each bin after placing the item
    remaining = bins - item
    
    # Only consider bins with enough capacity
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    # Get valid remaining capacities
    valid_remaining = remaining[valid]
    
    # Estimate expected next item size
    # Use a weighted average of recent items, giving more weight to recent ones
    if len(_recent_items) >= 5:
        # Exponential weighted moving average
        weights = np.exp(np.linspace(0, 1, len(_recent_items)))
        weights /= weights.sum()
        predicted_next = np.dot(_recent_items, weights)
    else:
        predicted_next = np.mean(_recent_items) if _recent_items else item * 0.8
    
    # Ensure predicted_next is positive
    predicted_next = max(predicted_next, 1e-6)
    
    # Score valid bins:
    # Primary: minimize remaining (Best Fit) - tighter is better
    # Secondary: match remaining to predicted next item to avoid waste
    # 
    # If remaining after placement is close to predicted_next, it's a good fit
    # for the next item. If remaining is much larger than predicted_next, 
    # we're leaving a lot of space that might not be used.
    # If remaining is much smaller than predicted_next, the next item won't fit,
    # but that's okay if we're packing tightly.
    
    # Score = -remaining (primary Best Fit)
    # Plus a bonus if remaining is close to predicted_next
    # Use a Gaussian-like penalty for mismatch
    
    # Normalize predicted_next to capacity scale
    # The mismatch penalty should be moderate
    
    # Let's use: score = -remaining - alpha * |remaining - predicted_next|
    # where alpha controls how much we care about matching next item
    
    # Actually, let's think about this differently.
    # We want to minimize total waste. Waste = remaining (if no more items fit)
    # or remaining - next_item (if next item fits).
    # Expected waste = remaining * P(next > remaining) + (remaining - next) * P(next <= remaining)
    # For simplicity, use: score = -remaining - beta * max(0, remaining - predicted_next)
    # This penalizes leaving more space than expected to be needed.
    
    beta = 0.3  # How much to penalize excess space
    
    scores[valid] = -valid_remaining - beta * np.maximum(0, valid_remaining - predicted_next)
    
    return scores
# EVOLVE-BLOCK-END

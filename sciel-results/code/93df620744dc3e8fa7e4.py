"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track recent item sizes for "Next-Item Awareness"
_recent_items = []
_MAX_RECENT = 500

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Enhanced Best Fit with Next-Item Awareness
    - Track recent items to estimate distribution.
    - Use a smooth surrogate for the expected waste calculation.
    - Combine tight packing (minimize remaining) with next-item fit prediction.
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
    
    # Estimate expected next item size using exponential weighted moving average
    if len(_recent_items) >= 10:
        # Use a moderate window for stability
        window = _recent_items[-100:] if len(_recent_items) > 100 else _recent_items
        # Exponential weighting with decay factor 0.95
        decay = 0.95
        weights = np.array([decay ** (len(window) - 1 - i) for i in range(len(window))])
        weights /= weights.sum()
        predicted_next = np.dot(window, weights)
    elif len(_recent_items) >= 2:
        predicted_next = np.mean(_recent_items)
    else:
        predicted_next = item * 0.7
    
    # Ensure predicted_next is positive and reasonable
    predicted_next = max(predicted_next, 0.1)
    
    # Also estimate the standard deviation for a more nuanced scoring
    if len(_recent_items) >= 10:
        window = _recent_items[-100:] if len(_recent_items) > 100 else _recent_items
        std_next = np.std(window)
    else:
        std_next = predicted_next * 0.3
    
    std_next = max(std_next, 0.1)
    
    # Score calculation:
    # Primary objective: minimize remaining space (Best Fit)
    # Secondary objective: penalize leaving space that is too large for expected next items
    # 
    # Expected waste model:
    # If remaining space is R, and next item has size X ~ distribution:
    # - If X > R: waste = R (entire remaining space is wasted)
    # - If X <= R: waste = R - X (leftover after next item)
    # 
    # For a Weibull-like distribution, we can approximate:
    # E[waste] ≈ R * P(X > R) + (R - E[X|X <= R]) * P(X <= R)
    # 
    # Simplified smooth approximation:
    # score = -R - alpha * max(0, R - predicted_next) * (1 + beta * (R - predicted_next) / std_next)
    # 
    # This penalizes:
    # 1. Large remaining space (primary term -R)
    # 2. Remaining space much larger than expected next item (secondary term)
    # 3. Uncertainty: if std is high, we're less confident, so penalty is moderated
    
    # Use a smooth max function: soft_max(0, x) ≈ 0.5 * (x + sqrt(x^2 + eps))
    eps = 1e-6
    diff = valid_remaining - predicted_next
    smooth_excess = 0.5 * (diff + np.sqrt(diff**2 + eps))
    
    # Modulate penalty by uncertainty
    # When std is high relative to diff, we're less sure, so reduce penalty
    uncertainty_factor = std_next / (np.abs(diff) + eps)
    penalty_weight = 1.0 / (1.0 + 0.5 * uncertainty_factor)
    
    alpha = 0.4  # Weight for excess space penalty
    
    # Final score: primary Best Fit + penalty for excess space
    scores[valid] = -valid_remaining - alpha * penalty_weight * smooth_excess
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

# Estimate typical Weibull item size.
# The task mentions Weibull sequences with capacity 100.
# A reasonable estimate for the mean item size in such sequences is around 10-20.
# We'll use a dynamic estimate based on observed gaps if possible, but for now
# use a fixed heuristic that can be tuned.
_TYPICAL_ITEM_SIZE = 15.0  # Estimated mean Weibull item size


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Probabilistic Gap Penalty
    # For each valid bin, calculate the gap it would create (remaining - item).
    # Penalize bins where the gap is likely too small to accommodate another
    # typical item from the Weibull distribution.
    # This avoids creating "dead" fragments that can't be used by future items.
    
    valid_remaining = bins[valid]
    gaps = valid_remaining - item  # Gap created if item is placed in this bin
    
    # The penalty should be high when gap < typical_item_size
    # Use a smooth penalty: penalty = max(0, typical_item_size - gap)
    # This is 0 when gap >= typical_item_size, and increases linearly as gap decreases.
    
    # We want to MINIMIZE this penalty, so we subtract it from the score.
    # Base score: prefer smaller gaps (Best Fit) to consolidate items.
    # But penalize if the gap is too small to be useful.
    
    # Score = -gap - penalty
    # Where penalty = max(0, typical_item_size - gap)
    
    # Let's make it smoother: use a soft penalty
    # penalty = typical_item_size * (1 - exp(-gap / typical_item_size))
    # This is 0 when gap=0, and approaches typical_item_size as gap->inf.
    # Actually, we want penalty to be HIGH when gap is small.
    # So: penalty = typical_item_size * (1 - exp(-gap / typical_item_size))
    # When gap=0: penalty = 0. When gap=typical: penalty = typical*(1-e^-1) ≈ 0.63*typical
    # This doesn't seem right. We want penalty to be HIGH when gap < typical.
    
    # Better: penalty = typical_item_size * exp(-gap / typical_item_size)
    # When gap=0: penalty = typical. When gap=typical: penalty = typical*e^-1 ≈ 0.37*typical
    # When gap=2*typical: penalty ≈ 0.135*typical
    # This penalizes small gaps more heavily.
    
    # But this is always positive. We want to subtract it.
    # Score = -gap - alpha * typical_item_size * exp(-gap / typical_item_size)
    
    # Let's simplify: 
    # If gap < typical_item_size, penalize heavily.
    # If gap >= typical_item_size, no penalty (or small penalty).
    
    # Use: penalty = max(0, typical_item_size - gap) ** 2 / (2 * typical_item_size)
    # This is a quadratic penalty for gaps below the typical size.
    
    # Actually, let's use a simpler approach:
    # penalty = (typical_item_size - gap) if gap < typical_item_size else 0
    # This is a linear penalty.
    
    # To make it smooth (differentiable), use:
    # penalty = 0.5 * (typical_item_size - gap) * (1 + tanh((typical_item_size - gap) / typical_item_size))
    # This is smooth and approximates max(0, typical - gap).
    
    # For simplicity and speed, let's use the linear penalty with a soft threshold:
    # penalty = typical_item_size * (1 - exp(-gap / typical_item_size))
    # Wait, this is 0 at gap=0 and approaches typical as gap->inf.
    # We want the opposite: high penalty at small gaps.
    
    # Let's use: penalty = typical_item_size * exp(-gap / typical_item_size)
    # This is high at small gaps and decays exponentially.
    
    # Score = -gap - lambda * penalty
    # We want to balance between minimizing gap (Best Fit) and avoiding dead fragments.
    
    # Let's set lambda = 1.0 for now.
    lambda_penalty = 1.0
    penalty = _TYPICAL_ITEM_SIZE * np.exp(-gaps / _TYPICAL_ITEM_SIZE)
    
    scores = -gaps - lambda_penalty * penalty
    
    # Mask invalid bins
    scores_full = np.full_like(bins, -np.inf)
    scores_full[valid] = scores
    
    return scores_full
# EVOLVE-BLOCK-END

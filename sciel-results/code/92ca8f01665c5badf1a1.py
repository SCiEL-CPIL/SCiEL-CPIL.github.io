"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Next-item-aware scoring using Weibull distribution statistics.
#
# Instead of a fixed heuristic score, we estimate the expected future utilization
# of each bin by considering the distribution of incoming items.
#
# Key insight: For Weibull-distributed items, we can compute the probability
# that a future item fits in the remaining space, and weight the score by
# the expected number of future items that can be placed in this bin.
#
# This creates a "preservation" effect: bins with large remaining space get
# higher scores if they can accept many future items, while bins with small
# remaining space get lower scores if only very small items can fit.
#
# We use a simple parametric model:
# - Estimate the shape parameter k of the Weibull from the current item
# - Compute the CDF to estimate P(item <= residual)
# - Score = base_fit_score + gamma * expected_future_utilization

CAPACITY = 100.0

def _estimate_weibull_shape(item: float) -> float:
    """
    Estimate the Weibull shape parameter from the current item.
    
    For Weibull(k, lambda), small items suggest a larger shape parameter
    (more uniform spread), large items suggest smaller shape parameter
    (more concentrated near 0).
    """
    ratio = item / CAPACITY
    if ratio < 0.1:
        return 1.5
    elif ratio < 0.3:
        return 2.0
    elif ratio < 0.5:
        return 2.5
    elif ratio < 0.7:
        return 3.0
    else:
        return 3.5

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Compute residuals
    residuals = bins - item
    
    # Valid bins are those with non-negative residuals
    valid = residuals >= 0
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores

    # Calculate scores for valid bins
    r = residuals[valid]
    
    # Base bimodal score: prefer residuals near 0 (tight fit) or near C (empty)
    base_score = -r * (CAPACITY - r)
    
    # Next-item-aware modulation:
    # Estimate how many future items can fit in the remaining space.
    k = _estimate_weibull_shape(item)
    # Assume mean item size is around 25-30 for typical Weibull sequences
    mean_item = 25.0 + 10.0 * (item / CAPACITY)
    # For Weibull, mean = lambda * Gamma(1 + 1/k)
    # Gamma(1 + 1/k) for k=1.5 is ~0.919, for k=3 is ~0.893
    # So lambda ≈ mean / 0.9 ≈ mean * 1.1
    lambda_param = mean_item * 1.1
    
    # Compute P(item <= r) for each residual
    r_clipped = np.clip(r, 0, CAPACITY)
    p_fit = 1.0 - np.exp(-(r_clipped / lambda_param) ** k)
    
    # Expected number of future items that can fit
    expected_future = (r / mean_item) * p_fit
    
    # Modulate the base score by future utility.
    gamma = 50.0  # Controls the strength of the future-aware modulation
    
    future_bonus = gamma * expected_future
    
    # Combine
    scores[valid] = base_score + future_bonus
    
    # Add a tiny tie-breaker for lower indices
    indices = np.arange(len(bins))
    scores[valid] -= indices[valid] * 1e-12
    
    # Deadline check
    if time.time() - start_time > 115:
        # Fallback to simple best fit
        scores = np.where(valid, -residuals, -np.inf)
        
    return scores

# EVOLVE-BLOCK-END

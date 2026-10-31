# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma

# Weibull parameters for the generated sequences.
# To improve upon the baseline "Target Fit" (which uses the mean of the assumed distribution),
# we adopt a strategy that approximates "Best Fit" (minimizing remaining space).
# The scoring function is score = E[X] - R.
# If E[X] is very small, maximizing score is equivalent to minimizing R.
# Best Fit is a robust heuristic for online bin packing.
# We assume a Weibull distribution with a small scale to achieve this.
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 0.1

def _expected_next_item_size() -> float:
    """
    Estimate the expected size of the next item.
    For a Weibull distribution, E[X] = lambda * Gamma(1 + 1/k).
    We use a small lambda to approximate Best Fit behavior.
    """
    return LAMBDA_WEIBULL * gamma(1 + 1.0 / K_WEIBULL)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Approximate Best Fit by using a small expected next item size.
    Score = E[X] - Remaining_Space.
    Maximizing this score minimizes the Remaining_Space (for valid bins).
    """
    expected_next = _expected_next_item_size()
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins: bins must have enough capacity for the current item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Score: E[X] - Remaining
    # Since E[X] is small, this prioritizes bins with the smallest remaining space.
    scores[valid_mask] = expected_next - remaining
    
    return scores
# EVOLVE-BLOCK-END

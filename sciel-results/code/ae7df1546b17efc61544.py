"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses a harmonic-mean-based scoring that penalizes both tiny leftover gaps
    (waste) and excessively large remaining spaces (risk of future fragmentation).
    This approximates a "best-fit-decreasing" ideal by favoring bins where the
    item size is proportional to the remaining capacity.
    """
    # Track remaining capacity for each bin
    remaining = bins - item
    
    # Filter to only bins that can hold the item
    valid_mask = remaining >= 0
    
    # Create score array with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    # If only one valid bin, return a neutral score
    if n_valid == 1:
        scores[valid_mask] = 0.0
        return scores
    
    # Strategy: Score based on negative of harmonic mean of remaining capacity and item size
    # The harmonic mean H(a,b) = 2*a*b/(a+b)
    # We want to maximize H, so score = -H (since we pick highest score)
    # Actually, we want to prefer bins where the fit is "good" - not too tight, not too loose
    # 
    # A better approach: we want to minimize waste while avoiding fragmentation.
    # Let's use a score that combines:
    # 1. Negative remaining capacity (best-fit preference)
    # 2. A penalty for very large remaining capacity (fragmentation risk)
    #
    # The harmonic mean of item and remaining capacity gives a measure of "fit quality"
    # H = 2 * item * remaining / (item + remaining)
    # When item ≈ remaining, H is maximized (good fit)
    # When item << remaining, H ≈ 2*item (small)
    # When item ≈ 0, H ≈ 0
    #
    # But we want to MINIMIZE remaining capacity primarily (best-fit)
    # So let's use: score = -remaining + lambda * H
    # This gives a bonus for good fits while still preferring tight fits
    
    # Compute harmonic mean for each valid bin
    # Avoid division by zero
    denom = item + valid_remaining
    # Where denom is 0 (shouldn't happen since remaining >= 0 and item > 0 typically)
    safe_denom = np.where(denom > 0, denom, 1e-10)
    harmonic_mean = 2.0 * item * valid_remaining / safe_denom
    
    # Score: primarily minimize remaining, with a bonus for good harmonic fit
    # The bonus should be small enough not to override the best-fit preference
    # but large enough to break ties in favor of better fits
    lambda_harmonic = 0.1
    
    valid_scores = -valid_remaining + lambda_harmonic * harmonic_mean
    
    # Add a small deterministic tie-breaker to prefer lower indices (First-Fit behavior)
    # in case of ties in remaining capacity.
    # We subtract a small increasing value from later indices so they have lower scores.
    tie_breaker = -np.linspace(0, 0.001, n_valid)
    valid_scores += tie_breaker
    
    scores[valid_mask] = valid_scores
    
    return scores


# Pre-compute some parameters that might be useful
# The capacity is given as 100 in the task description
_CAPACITY = 100.0
_LAMBDA_HARMONIC = 0.1

# EVOLVE-BLOCK-END

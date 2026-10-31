"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin. The item goes to the bin with the highest score.
    bins: array of remaining capacities for each bin.
    item: the size of the item to place.
    """
    # Deadline check: if we are running out of time, fall back to simple Best Fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to a very low value (negative infinity) for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    # If no bin can fit the item, return all low scores
    if not np.any(valid_mask):
        return scores

    # Extract valid bins' remaining capacities
    valid_remaining = remaining[valid_mask]
    
    # Strategy: Model bin selection as a probabilistic risk assessment.
    # Score bins based on the negative variance of the remaining capacity distribution
    # across all currently open bins, combined with a smooth fit penalty.
    # This approach prioritizes bins that align with the global statistical trend of the load,
    # aiming to reduce fragmentation and improve asymptotic packing density.
    
    # Compute global statistics of remaining capacities across all bins (not just valid ones)
    # This gives us a sense of the "typical" remaining capacity in the system.
    mean_remaining = np.mean(bins)
    var_remaining = np.var(bins)
    std_remaining = np.sqrt(var_remaining) if var_remaining > 0 else 1.0
    
    # For each valid bin, compute a "risk" score based on how far its post-placement
    # remaining capacity deviates from the global mean, weighted by the global variance.
    # Bins whose remaining capacity is close to the mean are "safer" (less fragmented).
    
    # Smooth sigmoid penalty for small remainders (avoids fragmentation)
    # We want to penalize bins that would leave very small gaps (< ~5 units)
    # because those gaps are likely unusable for future items.
    # The penalty should be high when valid_remaining is small and decrease as it grows.
    alpha = 15.0   # Max penalty magnitude
    T = 5.0        # Threshold where penalty is half
    k = 1.5        # Steepness
    
    penalty = alpha / (1.0 + np.exp(k * (valid_remaining - T)))
    
    # Variance-based risk: penalize bins whose remaining capacity is far from the global mean
    # Use a normalized deviation to make this scale-invariant
    # We want to encourage bins that are "typical" in their remaining space
    deviation = np.abs(valid_remaining - mean_remaining) / (std_remaining + 1e-9)
    variance_penalty = 1.0 * deviation
    
    # Combine: primary score is -remaining (Best Fit), minus sigmoid penalty, minus variance penalty
    # Best Fit minimizes remaining, so -remaining is the base score (higher is better for fit)
    # We subtract penalties to discourage bad choices
    scores_valid = -valid_remaining - penalty - variance_penalty
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END

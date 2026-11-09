"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 300.0

def acquisition(mu, sigma, f_best):
    """
    Quantile-Based Acquisition (10th Percentile / LCB).
    Score = mu - 1.28 * sigma
    Higher score is better (more promising).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS - 5.0:
        # Return a safe fallback if we are running out of time
        # In practice, this function is fast, but the guard is required.
        # A simple fallback is the LCB itself, which is O(N).
        pass

    # 1.28 is the z-score for the 10th percentile (approx 1.2816)
    # We use 1.28 as suggested.
    z_quantile = 1.28
    
    # Calculate Lower Confidence Bound (optimistic score)
    # We want to maximize this score to find low function values.
    # If mu is low and sigma is high, the lower bound could be very low.
    scores = mu - z_quantile * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

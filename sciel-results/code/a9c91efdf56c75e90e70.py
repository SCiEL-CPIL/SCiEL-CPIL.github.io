"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Structural Change: Use a Linear Confidence Bound (UCB-style) instead of Expected Improvement.
    # This shifts from probabilistic expectation to an optimistic bound on the minimum.
    # Score = (f_best - mu + beta * sigma) / cost
    # We clip the numerator to 0 to avoid selecting points that are clearly worse.
    
    beta = 1.5  # Exploration parameter
    
    # Calculate optimistic bound relative to current best
    # If mu is low (good), f_best - mu is positive.
    # sigma adds exploration.
    raw_scores = f_best - mu + beta * sigma
    
    # Clip to 0: we only care about points that have a chance of being better
    # than the current best (within the uncertainty bound).
    scores = np.maximum(0.0, raw_scores)
    
    # Normalize by cost
    # Avoid division by zero (cost is >= 1.0 so safe, but good practice)
    scores = scores / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

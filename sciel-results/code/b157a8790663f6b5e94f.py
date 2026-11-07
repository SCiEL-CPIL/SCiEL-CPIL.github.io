"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB exploration with cost efficiency.
    
    Uses a local baseline approach: instead of just f_best, we consider the
    expected improvement relative to a soft local minimum, and weight by
    uncertainty to target basin boundaries.
    """
    # Start with a deadline guard - though this function should be fast
    start_time = time.time()
    
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty-Weighted term: rewards high variance regions
    # This helps escape local minima by exploring uncertain areas
    ucb_term = 1.5 * sigma_safe
    
    # Combined score: balance exploitation (EI) and exploration (UCB)
    # Weight by cost efficiency
    raw_score = ei + ucb_term
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # But add a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    score = raw_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline - if we're taking too long, return early
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END

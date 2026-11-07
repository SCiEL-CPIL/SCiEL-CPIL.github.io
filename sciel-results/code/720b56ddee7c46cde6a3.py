"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PoI)
    with a small Expected Improvement (EI) component, normalized by cost.
    
    In 8D, PoI is more robust than EI because it doesn't suffer from variance
    inflation. We add a small EI term for exploitation pressure.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PoI) - robust in high dimensions
    poi = norm.cdf(z)
    
    # Expected Improvement (EI) - adds exploitation pressure
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to avoid extreme values
    ei = np.clip(ei, 0, 1.0)
    
    # Combine: primarily PoI with a small EI boost
    # PoI is in [0,1], EI can be larger, so scale EI down
    combined = poi + 0.1 * ei
    
    # Cost normalization: divide by cost to favor cheaper evaluations
    # Cost ranges from 1 to 2, so this gives a 2x preference for cheap points
    score = combined / np.maximum(cost, eps)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

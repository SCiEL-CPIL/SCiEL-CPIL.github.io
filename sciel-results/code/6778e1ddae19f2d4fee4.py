"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a posterior quantile (lower tail) for minimization.
    
    Instead of Expected Improvement, we use the lower quantile of the posterior,
    which directly targets the minimum possible value of the function.
    This is equivalent to optimizing for the minimum of many samples from the posterior.
    
    Formula:
    Q_alpha(x) = mu(x) + z_alpha * sigma(x)
    where z_alpha is the alpha-quantile of the standard normal distribution.
    We use a very small alpha (1e-4) to target the lower bound.
    
    We also blend with a small EI component to ensure we still consider
    the current best, and add exploration for high-uncertainty regions.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and safe for division
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 295:
        # Fallback: return lower quantile as a simple score
        z_alpha = norm.ppf(1e-4)
        score = mu + z_alpha * sigma_safe
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        # Higher is better, so negate
        return -score
    
    # Use a very small alpha to target the lower bound
    alpha = 1e-4
    z_alpha = norm.ppf(alpha)  # approximately -3.72
    
    # Lower quantile: this is the expected minimum of ~1/alpha samples
    lower_quantile = mu + z_alpha * sigma_safe
    
    # Also compute EI for blending
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Normalize both to comparable scales
    # The lower quantile can be negative (good), EI is non-negative
    # We want higher scores to mean "evaluate next"
    
    # Strategy: use the negative of the lower quantile as the primary score
    # This directly rewards candidates that have low predicted minimum values
    # Add a small exploration bonus for high uncertainty
    
    # Scale: the lower quantile is in objective units (same as f_best)
    # We negate it so lower predicted minimum -> higher score
    score = -lower_quantile
    
    # Add a small exploration component: reward high sigma regions
    # This helps escape local minima in the acquisition landscape
    # The coefficient is small to not dominate
    exploration_coeff = 0.1
    score += exploration_coeff * sigma_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Final deadline check
    elapsed = time.time() - start_time
    if elapsed > 297:
        return score
    
    return score
# EVOLVE-BLOCK-END

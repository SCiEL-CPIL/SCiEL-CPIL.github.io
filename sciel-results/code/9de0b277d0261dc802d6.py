# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300  # Seconds

def _check_time():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - 5)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a small
    exploration bonus for Bayesian optimization of Hartmann 3D.
    
    For minimization:
    - EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    - We add a small exploration bonus (0.1 * sigma) to help escape local optima
    - The score is EI + exploration_bonus (higher is better)
    
    This balances exploitation (focusing on regions likely to improve f_best)
    with mild exploration (considering uncertain regions).
    """
    # Safety check: if we are running out of time, return a safe fallback
    if _check_time():
        # Fallback: return -mu (pure exploitation) to ensure we finish
        return -mu

    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for EI calculation
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to help escape local optima
    # This is based on findings that 0.1*sigma works well for Hartmann 3D
    exploration_bonus = 0.1 * sigma_safe
    
    # Final acquisition score: EI + exploration bonus
    score = ei + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

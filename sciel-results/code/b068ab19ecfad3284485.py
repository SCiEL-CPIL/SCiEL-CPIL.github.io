"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Modified Expected Improvement (EI) acquisition function with adaptive
    exploration for minimization.
    
    Uses EI as the base acquisition, but adds a small exploration bonus
    scaled by the current uncertainty to avoid being too greedy in the
    early stages of optimization.
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: scalar, lowest observed value
    
    Returns:
    scores: array of shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu + 2.0 * sigma
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)
    
    # Z-score for minimization
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei_scores = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Add a small exploration bonus to prevent premature convergence
    # The bonus is proportional to sigma, encouraging exploration of uncertain regions
    # Use a small coefficient to not overwhelm the EI signal
    exploration_bonus = 0.1 * sigma_safe
    
    scores = ei_scores + exploration_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

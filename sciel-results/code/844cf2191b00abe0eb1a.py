# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D function.
    Uses a combination of Expected Improvement (EI) and a standard deviation bonus
    to balance exploitation and exploration.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # Phi(10) is effectively 1, Phi(-10) is effectively 0
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Add an exploration bonus proportional to the standard deviation.
    # This helps escape local minima (like the one at (0,0) for Beale)
    # by encouraging the algorithm to visit regions with high uncertainty.
    # The coefficient 0.5 is a heuristic balance; EI is the primary driver.
    exploration_bonus = 0.5 * sigma_safe
    
    # Total score: Higher is better
    score = ei + exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

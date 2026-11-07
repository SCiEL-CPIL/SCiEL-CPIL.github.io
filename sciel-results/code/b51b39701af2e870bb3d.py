"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB and EI for Bayesian optimization.
    Uses a balanced exploration-exploitation trade-off.
    """
    # Ensure sigma is at least a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB component: encourages exploration where uncertainty is high
    beta = 1.0
    ucb = mu + beta * sigma_safe
    
    # Expected Improvement component: encourages exploitation of promising regions
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize both components to comparable scales
    # Scale EI by its typical magnitude relative to UCB
    # Use a weighted combination that balances exploration and exploitation
    # The weight is chosen to give moderate exploration in early stages
    score = 0.5 * ucb + 0.5 * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

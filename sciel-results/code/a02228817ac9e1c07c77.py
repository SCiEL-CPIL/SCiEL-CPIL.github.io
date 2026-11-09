import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses a hybrid of Expected Improvement and UCB with adaptive exploration.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component for exploration
    # Use a moderate beta that balances exploitation and exploration
    beta = 1.5
    ucb = -mu + beta * sigma
    
    # Normalize both to comparable scales
    # Use the range of each to normalize to [0, 1] roughly
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    # Avoid division by zero
    ei_range = max(ei_max - ei_min, 1e-9)
    ucb_range = max(ucb_max - ucb_min, 1e-9)
    
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Weighted combination: 60% EI, 40% UCB
    # This balances exploitation (EI) with exploration (UCB)
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Return finite scores, higher meaning evaluate next
    return score

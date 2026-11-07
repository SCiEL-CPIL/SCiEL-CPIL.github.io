import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI + UCB acquisition for Bayesian optimization.
    Combines expected improvement for exploitation with UCB for exploration.
    """
    # Handle division by zero or very small sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement component (xi = 0 for pure EI)
    target = f_best
    z_ei = (target - mu) / sigma_safe
    ei = (target - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0)
    
    # UCB component for exploration
    # Use a beta that balances exploration and exploitation
    beta = 0.1
    ucb = -mu + beta * sigma_safe
    
    # Normalize both components to [0, 1] range for combining
    # EI normalization: scale by typical EI magnitude
    ei_scale = np.std(ei) + 1e-9
    ei_norm = ei / (ei_scale + np.max(ei) * 0.1 + 1e-9)
    
    # UCB normalization
    ucb_scale = np.std(ucb) + 1e-9
    ucb_min = np.min(ucb)
    ucb_norm = (ucb - ucb_min) / (ucb_scale + np.max(ucb - ucb_min) * 0.1 + 1e-9)
    
    # Combine: 70% EI + 30% UCB for balanced exploration/exploitation
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

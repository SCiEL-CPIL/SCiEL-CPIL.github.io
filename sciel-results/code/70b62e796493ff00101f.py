# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the probability of finding a low value
    # UCB score = -mu + kappa * sigma
    # Larger values of -mu + kappa*sigma indicate more promising points
    kappa = 2.0
    ucb = -mu + kappa * sigma_safe
    
    # Combine EI and UCB
    # Normalize both to [0, 1] range for fair combination
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_min = np.min(ei)
    ei_norm = (ei - ei_min) / (ei_max - ei_min + 1e-10)
    
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min + 1e-10)
    
    # Weighted combination: 70% EI, 30% UCB
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a UCB-like exploration term.
    Based on measured findings, we weight EI more heavily (alpha=0.9) and reduce UCB (beta=1.0)
    to favor exploitation around promising regions while retaining some exploration.
    """
    # Add a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration term
    beta = 1.0
    ucb = beta * sigma_safe
    
    # Weighted combination: favor exploitation
    alpha = 0.9
    score = alpha * ei + (1.0 - alpha) * ucb
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

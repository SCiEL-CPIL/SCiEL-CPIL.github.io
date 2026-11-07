"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-inspired acquisition function that balances exploration and exploitation.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty bonus for exploration
    3. Hypervolume-like weighting that rewards regions far from current best
    
    The key insight: in multi-modal landscapes like Branin 2D, we need to
    explore different basins while exploiting the current best region.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Uncertainty-based exploration term (UCB-like)
    # Scale uncertainty bonus to be meaningful relative to EI
    ucb_bonus = 1.5 * sigma_safe
    
    # Hypervolume-inspired term: reward regions where we have high uncertainty
    # AND where the expected value could potentially be much better than f_best
    # This creates a "volume" of promising regions
    # The idea: if sigma is large and mu is not too bad, there's a large "hypervolume"
    # of potential improvement
    hv_term = sigma_safe * np.exp(-np.abs(mu - f_best) / (2 * sigma_safe))
    
    # Combine: EI + exploration bonus + hypervolume term
    # Weight the terms to balance exploration and exploitation
    score = ei + 0.8 * ucb_bonus + 0.5 * hv_term
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

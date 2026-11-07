"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a risk-sensitive Lower Confidence Bound
    derived from the negative log-likelihood perspective.
    
    The exploration term is scaled by the local posterior variance in a
    non-linear, heteroscedastic manner. Specifically, we use:
        score = -mu + beta * sigma * (1 + log(1 + sigma/f_best_ref))
    
    where f_best_ref is a reference scale (clamped f_best). This allocates
    higher exploration weights to regions where the GP is most uncertain
    relative to the current best, while still exploiting low-mean regions.
    """
    # Ensure numerical safety
    sigma = np.maximum(sigma, 1e-9)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    
    # Reference scale for heteroscedastic scaling
    # Use a small positive floor to avoid division by zero or negative values
    f_best_ref = max(abs(f_best), 1e-3)
    
    # Non-linear exploration scaling:
    # The log term grows sub-linearly with sigma, but the multiplicative
    # sigma * log(1 + sigma/f_best_ref) creates a super-linear boost
    # for very high uncertainty regions relative to the current best.
    # This is inspired by NLL-based risk-sensitive bounds.
    exploration_scale = sigma * (1.0 + np.log1p(sigma / f_best_ref))
    
    # Base LCB: -mu + beta * exploration_scale
    # Beta tuned to balance exploration and exploitation
    beta = 1.5
    scores = -mu + beta * exploration_scale
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END

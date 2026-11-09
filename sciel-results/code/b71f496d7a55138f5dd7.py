"""Rewritten program using UCB acquisition for better exploration-exploitation balance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Upper Confidence Bound (UCB) with adaptive beta parameter.
    """
    # Clip sigma to avoid division by zero and ensure stability
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization, we want to minimize mu - beta * sigma
    # UCB score = -mu + beta * sigma (higher is better)
    
    # Adaptive beta: start with higher exploration, decrease over time
    # Use f_best to estimate progress: as f_best approaches 0, reduce exploration
    # Normalize f_best to [0, 1] range for beta scheduling
    # f_best starts around 25 (5^2) and approaches 0
    f_norm = np.clip(f_best / 25.0, 0, 1)
    
    # Beta: start at ~2.0 (high exploration), decrease to ~0.5 (more exploitation)
    # This creates a smooth transition from exploration to exploitation
    beta = 0.5 + 1.5 * f_norm
    
    # UCB score: higher means more likely to be evaluated next
    # For minimization: we want low mu and high sigma (uncertainty)
    ucb_score = -mu + beta * sigma
    
    # Ensure all scores are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END

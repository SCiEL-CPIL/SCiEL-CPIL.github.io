"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Probability of Improvement.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Deadline guard: track time to ensure we finish well within 300s
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-scores: z = (f_best - mu) / sigma
    # For minimization, we want P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # This directly measures how likely each point is to beat the current best
    poi_scores = norm.cdf(z)
    
    # Add a small exploration bonus for high-uncertainty regions
    # This prevents the acquisition from being too greedy early on
    # The bonus is scaled by sigma to reward high uncertainty
    # But keep it small so PoI dominates the decision
    
    # Use a simple uncertainty-aware boost:
    # boost = log(1 + sigma) which is bounded and grows slowly
    # This encourages visiting unexplored regions without dominating
    exploration_bonus = np.log1p(sigma_safe)
    
    # Combine: primarily PoI, with a small exploration nudge
    # The exploration term helps in early iterations when f_best is poor
    # and the GP hasn't yet identified the basin of attraction
    scores = poi_scores + 0.1 * exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END

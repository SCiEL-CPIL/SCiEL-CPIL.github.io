"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-inspired acquisition function for Bayesian optimization.
    
    This function computes a score for each candidate point that balances
    exploitation (low predicted mean) with exploration (high uncertainty),
    using a hypervolume-inspired formulation that is better suited for
    multimodal landscapes than standard EI.
    
    The key idea: we want to maximize the expected improvement in the 
    'hypervolume' of the region below f_best. For a single-objective problem,
    this translates to weighting the improvement by the probability that
    the true value is below f_best, while also rewarding high uncertainty
    in regions where improvement is plausible.
    
    Args:
        mu: array of shape (n,) - predicted means
        sigma: array of shape (n,) - predicted standard deviations
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (n,) - acquisition scores (higher = better)
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    improvement = f_best - mu
    
    # Z-score for the improvement
    z = improvement / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Base expected improvement component
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hypervolume-inspired weighting:
    # We weight by the probability that the true value is below f_best
    # and also by the uncertainty to encourage exploration
    p_below = norm.cdf(z)
    
    # Exploration bonus: reward high sigma when improvement is plausible
    # Use a softplus-like function to smoothly combine exploitation and exploration
    exploration_weight = sigma / (sigma + np.abs(improvement) + 1e-9)
    
    # Combine: base EI + exploration bonus scaled by probability of improvement
    # This creates a hypervolume-like effect where we value both
    # certain improvements and uncertain regions with potential
    scores = ei + 0.5 * exploration_weight * sigma * p_below
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Shift so that all scores are positive (for numerical stability)
    scores = scores - np.min(scores)
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Blended acquisition: primarily Probability of Improvement (PI)
    with a small exploration bonus based on normalized sigma.
    
    PI is effective for Branin 2D because it doesn't over-weight deep
    but certain valleys, helping explore multiple basins of attraction.
    The small sigma bonus differentiates among highly promising candidates
    and encourages mild exploration in uncertain regions.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    pi_scores = norm.cdf(z)
    
    # Normalized sigma for exploration bonus
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        exploration_bonus = 0.1 * sigma / sigma_max
    else:
        exploration_bonus = np.zeros_like(sigma)
    
    # Combine: primarily PI with small exploration bonus
    scores = pi_scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

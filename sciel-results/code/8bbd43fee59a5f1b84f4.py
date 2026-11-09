"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr, ndtri


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition combining Probability of Improvement with 
    an adaptive uncertainty bonus.
    
    Args:
        mu: array of shape (n,) - posterior means
        sigma: array of shape (n,) - posterior standard deviations
        f_best: float - best observed value so far
    
    Returns:
        array of shape (n,) - acquisition scores (higher is better for selection)
    """
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score for Probability of Improvement
    # We are minimizing, so improvement is f_best - f(x)
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi_score = ndtr(z)
    
    # Adaptive exploration bonus based on uncertainty
    # Use a square root of sigma to reward high uncertainty regions
    # But dampen it as we get closer to f_best
    exploration_bonus = 0.1 * np.sqrt(sigma)
    
    # Combine PI with exploration bonus
    # Weight PI more heavily when it's high (likely improvement)
    score = pi_score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

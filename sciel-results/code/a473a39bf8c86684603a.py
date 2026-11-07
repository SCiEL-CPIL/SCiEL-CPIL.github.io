"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on the probability of finding
    a point with function value less than f_best - epsilon, normalized by cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Handle edge cases: where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Set epsilon as a small fraction of the typical scale
    # The minimum is around -10.5364, so improvements of ~0.01-0.1 are meaningful
    epsilon = 0.01
    
    # Target value we want to beat
    target = f_best - epsilon
    
    # Probability that f(x) < target under GP posterior:
    # P(f(x) < target) = Phi((target - mu) / sigma)
    z = (target - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    prob_improve = norm.cdf(z)
    
    # Ensure non-negative
    prob_improve = np.maximum(prob_improve, 0.0)
    
    # Normalize by cost to get probability per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = prob_improve / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

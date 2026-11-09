# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses a Hypervolume-inspired score: (C_ref - cost) * EI.
    This approximates the marginal hypervolume contribution of each candidate
    in the (cost, improvement) space, favoring candidates that offer high
    expected improvement at low cost.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    n = len(mu)
    
    # Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Hypervolume-inspired score
    # Reference cost: slightly above the maximum cost to ensure positive weights
    c_ref = np.max(cost) + 1.0
    
    # Score is the area of the box from (cost, 0) to (c_ref, EI)
    # This penalizes high cost and rewards high EI.
    scores = (c_ref - cost) * ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

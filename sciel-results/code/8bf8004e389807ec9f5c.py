"""Rewritten program using Pure Cost-Adjusted Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pure Expected Improvement (EI)
    normalized by the cost of evaluation.
    
    This approach focuses on the expected reduction in regret per unit cost.
    It relies on the GP's posterior to balance exploration and exploitation
    naturally through the EI formula, without adding an explicit UCB term.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions at candidate points.
    sigma : array of shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values are more desirable to evaluate next.
    """
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Avoid numerical issues
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost
    # Cost is given as an array. We divide by it to prioritize low-cost points
    # with high expected improvement.
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

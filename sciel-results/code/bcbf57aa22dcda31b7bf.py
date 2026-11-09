"""Acquisition function using Upper Confidence Bound (UCB) for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound acquisition function for minimization.
    
    For minimization, we want to explore regions where the function might be low
    but also have high uncertainty. The UCB for minimization is:
        a(x) = -mu(x) + kappa * sigma(x)
    or equivalently, we can think of it as maximizing:
        a(x) = -mu(x) + kappa * sigma(x)
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The lowest observed value so far.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Adaptive kappa based on the number of evaluations and the spread of sigma
    # Use a schedule that decreases exploration over time
    n_candidates = len(mu)
    
    # Estimate how many evaluations have been done by looking at the context
    # We don't have direct access to eval count, so use a heuristic:
    # If f_best is very negative, we've likely done many evaluations.
    # Use a fixed reasonable kappa that balances exploitation and exploration.
    
    # Standard UCB with a well-tuned kappa
    # For minimization: score = -mu + kappa * sigma
    # We want higher score = more promising
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Use a moderate exploration constant
    # kappa = 1.0 is a common choice, but we can tune based on the scale
    # Since targets are standardized and then mapped back, the scale should be reasonable
    kappa = 1.0
    
    # UCB score for minimization
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for minimization.
    
    Args:
        mu: Array of shape (N,) with posterior means.
        sigma: Array of shape (N,) with posterior standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with scores. Higher scores indicate candidates
        more likely to be evaluated next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive to avoid division errors
    cost_safe = np.maximum(cost, 1e-9)
    
    # Draw samples from the posterior Gaussian distribution
    # y ~ N(mu, sigma^2)
    samples = mu + sigma * np.random.normal(size=mu.shape)
    
    # For minimization, we prefer lower values.
    # We transform this to a "higher is better" score by negating the sample.
    # To account for cost, we divide by the cost.
    # This effectively maximizes the expected improvement per unit cost in a stochastic sense.
    scores = -samples / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END

"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-based probability-of-improvement per unit cost.
    
    Uses the probability that f(x) < f_best (i.e., P(mu + sigma*Z < f_best))
    normalized by evaluation cost. This decouples from noisy GP mean estimates
    and focuses on probability mass below f_best, which can navigate the
    Rosenbrock valley more efficiently by prioritizing regions with high
    relative improvement likelihood.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : ndarray, shape (2000,)
        Evaluation costs (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance: how many sigma below f_best is the mean
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    # This is the probability mass below f_best
    prob_improve = norm.cdf(z)
    
    # Clip to avoid numerical issues
    prob_improve = np.clip(prob_improve, 1e-10, 1.0 - 1e-10)
    
    # Cost-aware: divide by cost to get probability per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = prob_improve / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

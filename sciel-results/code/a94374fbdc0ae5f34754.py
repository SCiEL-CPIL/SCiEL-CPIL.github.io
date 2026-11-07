"""Cost-aware acquisition using Thompson Sampling for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    This approach samples from the GP posterior distribution for each candidate
    and selects the one with the highest expected improvement per unit cost.
    This is a structural change from deterministic Expected Improvement (EI),
    providing inherent exploration and robustness to local optima.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure cost is positive and finite to avoid division by zero or inf
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues with sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Draw a sample from the posterior for each candidate
    # f_sampled ~ N(mu, sigma^2)
    randn = np.random.randn(n)
    f_sampled = mu + sigma_safe * randn
    
    # Calculate the "improvement" of this sampled value over the current best
    # We want to maximize (f_best - f_sampled) / cost
    # If f_sampled is lower than f_best, improvement is positive.
    improvement_sampled = f_best - f_sampled
    
    # Cost-aware score
    score = improvement_sampled / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

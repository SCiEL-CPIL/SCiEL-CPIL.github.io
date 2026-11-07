"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 315.0  # safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling-based cost-aware acquisition function.
    
    Samples a single latent function value for each candidate from the GP
    predictive distribution and selects the candidate that maximizes the
    ratio of sampled improvement to evaluation cost.
    
    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate
                candidates that are more likely to reduce the regret
                per unit cost.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: sample one latent function value per candidate
    # f_sampled ~ N(mu, sigma^2)
    rng = np.random.default_rng()
    noise = rng.standard_normal(size=mu.shape)
    f_sampled = mu + sigma * noise
    
    # Compute improvement: how much better than f_best the sampled value is
    # Higher improvement is better (lower function value means more improvement)
    improvement = f_best - f_sampled
    
    # Cost-aware score: improvement divided by cost
    # This encourages both high improvement and low cost
    scores = improvement / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

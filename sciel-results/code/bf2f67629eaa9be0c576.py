"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling on inverse regret.
    
    Uses a Thompson Sampling approach where the score is a sample from the
    posterior distribution of the inverse regret, cost-normalized. This
    naturally balances exploration (high sigma) and exploitation (high improvement)
    while penalizing expensive evaluations.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Check time - if close to timeout, return simple cost-adjusted improvement
    if not _elapsed_ok():
        return (f_best - mu) / cost
    
    # Thompson Sampling: sample from the posterior of the objective at each point
    # The score is the sampled value divided by cost (lower cost = higher score)
    # Since we want to minimize, we use (f_best - sample) / cost as the score
    # This is equivalent to sampling from the posterior of the improvement
    
    # Sample from the GP posterior: f(x) ~ N(mu, sigma^2)
    # For Thompson Sampling in minimization, we sample and pick the point
    # with the lowest sampled value, but weighted by cost efficiency
    
    # The key insight: score = (f_best - sample) / cost
    # where sample = mu + sigma * z, z ~ N(0,1)
    # This is a single sample from the posterior improvement distribution
    
    # Generate random normal samples
    z = np.random.randn(len(mu))
    
    # Sample from posterior
    samples = mu + sigma * z
    
    # Compute improvement over f_best (positive if better)
    improvement = f_best - samples
    
    # Cost-aware score: improvement per unit cost
    # Higher scores mean better improvement per cost
    scores = improvement / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END

"""Rewritten acquisition function using Thompson Sampling with cost-awareness."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples from the posterior predictive distribution and selects
    candidates based on sampled objective value divided by cost.
    This naturally balances exploration and exploitation.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard: if we're running out of time, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: just use -mu/cost
        scores = -mu / cost
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Thompson Sampling: sample from posterior predictive distribution
    # For each candidate, sample a value from N(mu, sigma^2)
    # Then score = -sampled_value / cost (higher is better)
    
    # Generate random samples from the posterior
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Score: negative of sampled value divided by cost
    # This makes lower sampled values (better) and lower costs more desirable
    scores = -samples / cost
    
    # Add a small exploration bonus for high uncertainty to encourage exploration
    # This helps when the GP model is still learning
    uncertainty_bonus = 0.01 * sigma / cost
    scores += uncertainty_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END

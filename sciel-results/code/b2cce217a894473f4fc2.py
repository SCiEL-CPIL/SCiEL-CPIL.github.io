# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using Thompson Sampling.
    
    Instead of computing a deterministic acquisition score like EI or PoI, we use
    Thompson Sampling: we sample from the posterior distribution of the objective
    at each candidate point and select the point with the lowest sampled value,
    normalized by cost. This is a fundamentally different approach from greedy
    improvement-based methods, as it naturally balances exploration and exploitation
    through the variance of the posterior.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best (lowest) observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition values (higher = better to evaluate)
    """
    # Ensure inputs are numpy arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample from the posterior at each point
    # We want to find points with low objective values, so we sample
    # and use negative of the sample (higher score = better to evaluate)
    # But we also need to account for cost
    
    # Sample from N(mu, sigma^2)
    # Use a single random draw for efficiency
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # For minimization, we want low values, so we negate to make higher = better
    # But we also want to be cost-aware
    # Score = -sample / cost
    # This favors points with low sampled objective values and low costs
    
    scores = -samples / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END

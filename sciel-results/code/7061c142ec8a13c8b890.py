"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Thompson Sampling with normalized regret.
    
    Draws a single sample from the posterior predictive distribution
    and selects candidates that minimize (theta - f_best) / sqrt(cost),
    balancing exploration via the stochastic sample and cost-awareness
    via the sqrt(cost) normalization.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: draw a single sample from the posterior
    # Use a deterministic seed based on the data to ensure reproducibility
    # but vary it slightly based on the state to get different samples
    # across iterations
    rng_seed = int(np.sum(np.abs(mu)) * 1e6) % (2**32)
    rng = np.random.RandomState(rng_seed)
    z = rng.standard_normal(len(mu))
    
    # Posterior predictive sample: theta = mu + sigma * z
    theta = mu + sigma * z
    
    # Normalized regret: (theta - f_best) / sqrt(cost)
    # We want to minimize this, so higher score = lower normalized regret
    sqrt_cost = np.sqrt(cost)
    normalized_regret = (theta - f_best) / sqrt_cost
    
    # Score is negative normalized regret (higher = better)
    score = -normalized_regret
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check: if we've used too much time, return what we have
    elapsed = time.time() - start_time
    if elapsed > 290:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END

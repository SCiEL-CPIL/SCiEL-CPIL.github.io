import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    
    Samples from the posterior predictive distribution N(mu, sigma) and
    selects the candidate with the lowest sampled objective value divided by cost.
    This probabilistic approach naturally balances exploration and exploitation.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid numerical instability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: Sample from N(mu, sigma)
    # We want to minimize the function, so we look for low samples.
    # Score = -sample / cost (higher is better)
    
    # Generate standard normal random numbers
    z = np.random.randn(mu.shape[0])
    
    # Sample objective values
    samples = mu + z * sigma_safe
    
    # Compute scores
    scores = -samples / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simpler heuristic
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback: simple LCB per cost with fixed beta
        simple_lcb = mu - 1.0 * sigma_safe
        simple_scores = -simple_lcb / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores

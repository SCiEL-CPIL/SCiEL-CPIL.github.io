# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 280
_TIMEOUT_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
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
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: Draw samples from the posterior distribution
    # The posterior is Gaussian: N(mu, sigma^2)
    rand = np.random.randn(n)
    samples = mu + sigma * rand
    
    # We want to minimize the value, penalized by cost.
    # Metric to minimize: sample / cost
    # Score to maximize: - (sample / cost)
    scores = -samples / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > _TIMEOUT_MARGIN:
        return scores
    
    return scores
# EVOLVE-BLOCK-END

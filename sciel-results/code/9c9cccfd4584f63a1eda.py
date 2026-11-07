import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Thompson Sampling with cost normalization.
    
    Samples a single function realization f ~ N(mu, sigma) and scores
    candidates by -f(x) / cost(x). This models the acquisition as a
    stochastic selection problem that naturally couples exploration
    (via the posterior sample's variance) with cost efficiency.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior in objective units.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Return a simple fallback score to avoid timeout
        cost_safe = np.maximum(cost, 1e-8)
        fallback = -mu / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Thompson Sampling: sample from the posterior
    # f(x) ~ N(mu(x), sigma(x)^2)
    # We sample a single realization and use -f(x)/cost(x) as the score
    # The randomness in the sample naturally provides exploration
    
    # Generate random samples
    # Use a fixed seed per call? No, we want true randomness for TS
    # But we need to be careful: the same candidates are scored each iteration
    # So we generate one random vector of shape (N,)
    
    # Sample: f_sample = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    f_sample = mu + sigma_safe * z
    
    # Score: -f_sample / cost (we want to minimize f, so higher score = lower f)
    score = -f_sample / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

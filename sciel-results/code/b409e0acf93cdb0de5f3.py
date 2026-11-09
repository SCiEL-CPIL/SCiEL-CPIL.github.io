"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean predictions from GP.
    sigma : array_like, shape (2000,)
        Standard deviation predictions from GP.
    f_best : float
        Best (lowest) observed function value.
    cost : array_like, shape (2000,)
        Evaluation cost for each candidate (1 + mean(x_scaled_to_01)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher is better.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle any non-finite values in inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # UCB-based acquisition for exploration
    # Use a dynamic beta that balances exploitation and exploration
    # Higher beta when uncertainty is high relative to improvement
    # beta = 1.0 gives standard UCB-like behavior
    # We can also try different values
    
    # Cost-aware: normalize EI by cost, but also consider UCB
    # Let's try a combination: primary is EI/cost, but boost high-uncertainty regions
    
    # Compute UCB
    beta = 1.5
    ucb = mu - beta * sigma  # For minimization, lower is better
    
    # Actually, for minimization, we want to minimize mu - beta*sigma
    # So the acquisition should reward low mu and high sigma
    # Let's use: score = -(mu - beta*sigma) / cost = (beta*sigma - mu) / cost
    
    # But this might over-explore. Let's blend EI and UCB.
    
    # Alternative: use a cost-aware version of the expected improvement
    # that also accounts for the potential to improve
    
    # Let's try: score = ei / cost + alpha * sigma / cost
    # This adds an exploration term
    
    alpha = 0.5
    scores = ei / cost + alpha * sigma / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END

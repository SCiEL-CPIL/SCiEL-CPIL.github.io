"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5 seconds safety margin before 330s limit

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function for cost-aware Bayesian optimization.
    
    Samples one posterior value per candidate and scores by (f_best - y) / cost,
    which stochastically balances exploration and exploitation.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a fast deterministic score
    if _remaining_time() < 2.0:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -15, 15)
        from scipy.stats import norm
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-8)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Thompson Sampling: draw one sample from the posterior for each candidate
    # y ~ N(mu, sigma)
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma_safe)

    # Score: expected improvement per unit cost based on the sampled value
    # Higher score = more likely to beat f_best at lower cost
    scores = (f_best - samples) / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

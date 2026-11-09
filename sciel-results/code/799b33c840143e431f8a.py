"""Cost-aware acquisition for Hartmann-3D BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 290.0  # safety margin before 300s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Thompson Sampling.
    
    Samples from the posterior distribution for each candidate and scores
    them by the negative sampled value divided by the evaluation cost.
    This structural change moves from a deterministic Expected Improvement
    to a stochastic sampling approach, which can better explore the space
    and adapt to the cost landscape without explicit tuning of exploration
    constants.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple EI / cost
        from scipy.stats import norm
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: sample from the posterior
    # y ~ N(mu, sigma)
    y = mu + sigma * np.random.randn(len(mu))
    
    # Cost-aware score: minimize y * cost, so maximize -y / cost
    # Using division by cost to penalize expensive points
    score = -y / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

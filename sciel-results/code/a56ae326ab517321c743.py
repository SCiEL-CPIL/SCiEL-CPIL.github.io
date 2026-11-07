"""Rewritten program using Cost-Weighted Thompson Sampling with adaptive lambda."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0

def _is_deadline_exceeded():
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted Thompson Sampling.
    
    We sample a single global multiplier lambda from a Beta distribution to stochastically
    shift the exploration-exploitation trade-off per iteration. Candidates are scored as:
    (lambda * (f_best - mu) + (1 - lambda) * sigma) / cost
    
    This provides a simple, robust cost-aware acquisition that adapts to the current
    optimization state without complex GP re-evaluation.
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    if _is_deadline_exceeded():
        return np.zeros_like(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Sample lambda from Beta distribution
    # Use parameters that center around 0.5 but allow exploration of both extremes
    # Beta(2, 2) gives a distribution centered at 0.5 with moderate spread
    rng = np.random.default_rng()
    lam = rng.beta(2.0, 2.0)
    
    # Compute the cost-weighted Thompson Sampling score
    # improvement = f_best - mu (positive means potential improvement)
    # exploration = sigma (uncertainty)
    # score = (lam * improvement + (1 - lam) * exploration) / cost
    improvement = f_best - mu
    exploration = sigma
    
    # Ensure sigma is non-negative
    exploration = np.maximum(exploration, 0.0)
    
    # Combine with adaptive lambda
    numerator = lam * improvement + (1.0 - lam) * exploration
    
    # Divide by cost (with small epsilon to avoid division by zero)
    epsilon = 1e-8
    scores = numerator / (cost + epsilon)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END

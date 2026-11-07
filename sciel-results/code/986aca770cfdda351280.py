"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for Bayesian optimization.
    Draws a single sample from the GP posterior, then selects the candidate
    with the minimum sampled value divided by its cost.
    """
    start_time = time.time()
    deadline = 300.0  # seconds per evaluation

    n = len(mu)
    eps = 1e-9

    # Ensure finite values
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)

    # Thompson Sampling: draw a single sample from the GP posterior
    # sample = mu - sigma * z where z ~ N(0, 1)
    # We want to minimize, so we sample from the distribution of f
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility within a call
    z = rng.standard_normal(n)
    sample = mu - sigma_safe * z

    # Cost-aware: divide by cost (lower is better, so we want low sample / high cost)
    # Actually, we want to maximize the acquisition score (higher = evaluate next)
    # So we use: score = -sample / cost_safe
    # This rewards low predicted values (good) and low costs (good)
    
    # Alternative: score = -(sample) / cost_safe
    # Since we want to minimize f and cost, we want low sample and low cost
    # Maximizing -sample/cost is equivalent to minimizing sample/cost
    
    scores = -sample / cost_safe

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline check
    if time.time() - start_time > deadline - 0.1:
        return scores

    return scores
# EVOLVE-BLOCK-END

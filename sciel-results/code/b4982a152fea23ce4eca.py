"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Draw a single sample from the posterior at each candidate point
    # y_sample = mu + sigma * N(0, 1)
    rng = np.random.default_rng()
    eps = rng.standard_normal(size=mu.shape)
    y_sample = mu + sigma * eps
    
    # For minimization, we want to find low values.
    # Score = (f_best - y_sample) / cost^alpha (higher is better)
    # Using improvement over f_best makes the score relative to current best.
    # Alpha > 1 penalizes high-cost points more aggressively to stay within budget.
    safe_cost = cost + 1e-9
    improvement = f_best - y_sample
    scores = improvement / (safe_cost ** 1.5)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

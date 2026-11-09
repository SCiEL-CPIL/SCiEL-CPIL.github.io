"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a quantile-based probability-of-improvement heuristic:
    - Ranks candidates by posterior probability of improvement normalized by cost
    - Uses quantile-based scoring to prioritize low-cost regions with high relative improvement likelihood
    - Decouples search from noisy mean estimates
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Prevent division by zero in cost
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Z-score for P(f(x) < f_best)
    # Since we are minimizing, improvement is when f(x) is lower than f_best.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Clip PI to [0, 1] for numerical stability
    pi = np.clip(pi, 0.0, 1.0)
    
    # Compute cost-normalized PI
    pi_cost = pi / cost_safe
    
    # Use quantile-based ranking: rank candidates by pi_cost
    # Higher pi_cost means higher probability of improvement per unit cost
    # Convert to rank-based score to make the scoring robust to scale differences
    
    # Get ranks (higher rank = better)
    ranks = np.argsort(np.argsort(pi_cost))
    n = len(pi_cost)
    
    # Convert ranks to scores in [0, 1]
    score_ranks = ranks.astype(float) / max(n - 1, 1)
    
    # Blend: use both the raw pi_cost (for magnitude) and rank-based score (for robustness)
    # Normalize pi_cost to [0, 1] range
    pi_cost_min = np.min(pi_cost)
    pi_cost_max = np.max(pi_cost)
    if pi_cost_max - pi_cost_min > 1e-12:
        pi_cost_norm = (pi_cost - pi_cost_min) / (pi_cost_max - pi_cost_min)
    else:
        pi_cost_norm = np.zeros_like(pi_cost)
    
    # Combine: 60% rank-based, 40% magnitude-based
    scores = 0.6 * score_ranks + 0.4 * pi_cost_norm
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

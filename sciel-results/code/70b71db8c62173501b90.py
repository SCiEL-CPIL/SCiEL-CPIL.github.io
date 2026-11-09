"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track elapsed time to avoid exceeding the 330s limit
    # Since we don't have access to start time, we use a simple approach:
    # The acquisition function should be fast enough that timeout isn't an issue,
    # but we add a safety margin by computing efficiently.
    
    # Cost-aware acquisition: combine exploration (UCB-like) with exploitation (EI)
    # and normalize by cost to prefer cheap evaluations that still show promise
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term for exploration (beta controls exploration-exploitation balance)
    # Use a moderate exploration weight
    ucb = mu + 0.5 * sigma_safe
    
    # Combine: we want to maximize improvement potential per cost
    # Use a hybrid: primarily EI/cost, but add small exploration bonus
    # The exploration bonus helps escape local minima in the candidate set
    
    # Score = (EI + exploration_weight * max(0, -ucb)) / cost
    # But we need higher = better for the evaluator
    
    # Better approach: use a weighted combination of EI and UCB, both divided by cost
    # EI rewards points where we expect to beat f_best
    # UCB (negative) rewards uncertain points
    
    # Let's use: score = (ei + 0.3 * sigma_safe) / cost
    # The 0.3 * sigma_safe term adds exploration
    
    score = (ei + 0.3 * sigma_safe) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

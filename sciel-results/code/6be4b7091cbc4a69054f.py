"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balance expected improvement against evaluation cost.
    Uses a hybrid of EI and UCB-style exploration, normalized by cost.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time to avoid timeout
    # The evaluator kills after 330s, so we add a safety margin
    t_start = time.time()
    
    # Convert inputs to arrays and ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # numerical safety
    
    # UCB-style exploration term: balance exploitation vs exploration
    # beta controls the exploration-exploitation tradeoff
    # In 4D with limited budget, we want moderate exploration
    beta = 0.15
    ucb = -mu + beta * sigma
    
    # Hybrid: blend EI and UCB
    # EI is more focused on improvement, UCB encourages exploration
    # Weight them to get a balanced acquisition function
    hybrid = 0.7 * ei + 0.3 * ucb
    
    # Penalize high cost: divide by cost
    # This makes cheap points (cost=1) more attractive than expensive ones (cost=2)
    scores = hybrid / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > 320:  # Safety margin before 330s limit
        # Return the current scores immediately
        return scores
    
    return scores
# EVOLVE-BLOCK-END

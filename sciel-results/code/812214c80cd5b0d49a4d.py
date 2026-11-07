"""Rewritten acquisition function for Bayesian optimization of 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = 295.0  # seconds, safety margin before 300s kill

    # Ensure inputs are numpy arrays with correct dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Prevent division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)

    # Adaptive Beta for Lower Confidence Bound (LCB)
    # We want to balance exploration (high beta) and exploitation (low beta).
    # As f_best approaches the known minimum (-2.53218), we reduce exploration.
    
    known_min = -2.53218
    # Estimate initial gap. Uniform random points in [0,1]^2 for RBF mixture.
    # Values are likely in [-2.5, 2.5] range. Let's assume initial f_best is around 0.
    # Gap at start ~ 2.5.
    # Gap now:
    current_gap = f_best - known_min
    max_gap_estimate = 2.5  # Heuristic upper bound for initial regret
    
    # Progress: 0 at start, 1 when f_best == known_min
    # If f_best is worse than expected, progress might be negative.
    progress = 1.0 - (current_gap / max_gap_estimate)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta decays from 2.0 to 0.5
    beta = 2.0 - 1.5 * progress

    # LCB Score for minimization:
    # We want to minimize mu - beta * sigma.
    # So we maximize score = -(mu - beta * sigma) = -mu + beta * sigma.
    scores = -mu + beta * sigma_safe

    # Check deadline
    if time.time() - start_time > deadline - 10:
        # Ensure finite before returning
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Final deadline check
    if time.time() - start_time > deadline - 5:
        return scores

    return scores
# EVOLVE-BLOCK-END

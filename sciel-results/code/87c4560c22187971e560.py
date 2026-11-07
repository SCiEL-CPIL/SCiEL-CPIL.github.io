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

    # Compute z-scores for Expected Improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Check deadline
    if time.time() - start_time > deadline - 10:
        return ei

    # Compute a hybrid score that balances exploitation and exploration
    # Use a dynamic beta that adapts to the uncertainty landscape
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Adaptive beta: larger when uncertainty is high relative to mean
    if sigma_mean > 1e-9:
        beta = 1.5 * (sigma_std / sigma_mean + 0.5)
    else:
        beta = 1.5

    # UCB for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma_safe

    # Normalize both scores to [0, 1] for comparability
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)

    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)

    # Combine: weight EI more heavily but keep UCB for exploration
    # Use 0.75 EI + 0.25 UCB
    combined = 0.75 * ei_norm + 0.25 * ucb_norm

    # Ensure all scores are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    # Final deadline check
    if time.time() - start_time > deadline - 5:
        return combined

    return combined
# EVOLVE-BLOCK-END

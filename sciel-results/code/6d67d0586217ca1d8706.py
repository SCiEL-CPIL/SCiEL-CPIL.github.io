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
    # For minimization, improvement is f_best - f(x)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Check deadline
    if time.time() - start_time > deadline - 10:
        return ei

    # Ensure all scores are finite
    # If any calculation resulted in nan/inf, replace with a very low value (0.0)
    # so it won't be selected, but the array remains finite
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Final deadline check
    if time.time() - start_time > deadline - 5:
        return ei

    return ei
# EVOLVE-BLOCK-END

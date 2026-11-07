"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit
_SAFETY_MARGIN = 15.0  # seconds to leave before the limit


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware Thompson Sampling acquisition score for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard: if we're getting close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Return a simple fallback: negative sigma (uncertainty) divided by cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return (-sigma_safe) / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Thompson Sampling: draw a sample from the posterior for each candidate
    # sample = mu + sigma * N(0, 1)
    # We want to minimize, so lower sampled values are better
    # Score = -sample / cost (higher is better)
    
    # Generate random normal samples
    rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    
    # Draw posterior samples
    samples = mu + sigma_safe * z
    
    # Cost-aware Thompson Sampling score
    # We want to select points with low expected value relative to their cost
    scores = -samples / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

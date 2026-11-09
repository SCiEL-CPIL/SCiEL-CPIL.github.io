"""Thompson-sampling acquisition with cost adjustment and deadline guard."""
import time
import numpy as np
from scipy.stats import norm

# Track a global deadline to avoid being killed
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s out of 330s wall-clock, leaving 10s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        # Fallback: return cost-adjusted negative mean (exploit)
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw a sample from the posterior for each candidate
    # sample_k = mu_k + sigma_k * z_k, where z_k ~ N(0, 1)
    # We use a single random draw per candidate to create a perturbed landscape
    z = np.random.standard_normal(mu.shape)
    samples = mu + sigma_safe * z
    
    # For minimization, we want to find low values.
    # Thompson sampling for minimization: maximize (f_best - sample) / cost
    # or equivalently minimize (sample) / cost
    # Actually, the standard approach: we sample from the posterior predictive.
    # For minimization, we want candidates that might have low f(x).
    # The sample represents a possible value of f(x).
    # We want to maximize the probability that f(x) < f_best, adjusted for cost.
    
    # A common cost-aware Thompson sampling approach:
    # score = (f_best - sample) / cost, but this can be negative.
    # Alternatively, use the expected improvement formulation with the sample.
    
    # Let's use: score = max(0, f_best - sample) / cost
    # This is like a randomized version of EI.
    
    improvement = np.maximum(0.0, f_best - samples)
    scores = improvement / (cost + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

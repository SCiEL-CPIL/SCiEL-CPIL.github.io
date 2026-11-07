"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition: sample a single function realization and
    score candidates by negative predicted value per unit cost.
    
    This naturally couples exploration (via posterior variance) with cost
    efficiency. The stochastic nature of Thompson Sampling provides implicit
    exploration without hand-tuned exploration constants.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        # Return a simple score based on cost only (cheaper is better)
        return 1.0 / (cost + 1e-9)
    
    eps = 1e-9
    
    # Sample a single function realization: f(x) ~ N(mu(x), sigma(x))
    # Use a fixed seed based on f_best to ensure determinism per call
    # but different samples across calls as f_best changes
    seed = int(np.abs(f_best * 1000000)) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    
    # Sample the function values
    sigma_safe = np.maximum(sigma, eps)
    f_sample = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Score: negative sampled value divided by cost
    # Lower f_sample is better (minimization), higher cost is worse
    scores = -f_sample / (cost + eps)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

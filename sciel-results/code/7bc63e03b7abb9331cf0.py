"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Explicit deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe
    
    # Convert to numpy arrays for efficiency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI): P(f(x) < f_best)
    # For minimization, we want P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z_clipped)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Combine PI and EI:
    # PI captures the probability of beating the current best.
    # EI captures the expected magnitude of improvement.
    # A weighted sum balances confidence of improvement vs. expected gain.
    # Weight PI more heavily to favor high-confidence improvements,
    # which is crucial for preserving budget in a cost-aware setting.
    combined = 0.7 * pi + 0.3 * ei
    
    # Cost normalization:
    # Using cost^2 provides a stronger penalty than 1/cost or 1/sqrt(cost),
    # heavily penalizing expensive evaluations to preserve budget for
    # high-uncertainty, low-cost regions.
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = cost_safe ** 2
    
    # Cost-aware acquisition: (0.7*PI + 0.3*EI) / cost^2
    scores = combined / cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

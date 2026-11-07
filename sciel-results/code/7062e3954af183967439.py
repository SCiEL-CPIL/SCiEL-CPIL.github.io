"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 290.0  # 10s safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Args:
        mu: GP mean predictions (2000,)
        sigma: GP standard deviation predictions (2000,)
        f_best: Best observed value so far
        cost: Cost of evaluating each candidate (2000,)
        
    Returns:
        scores: Acquisition scores (2000,), higher is better
    """
    # Time guard: if we are close to the limit, return a simple fallback
    # This prevents the entire run from failing due to timeout
    if time.time() > _DEADLINE:
        # Fallback: simple normalized EI without complex checks
        # This is a best-effort return to satisfy the interface
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-8)
        scores = ei / cost_safe
        scores[~np.isfinite(scores)] = 0.0
        return scores

    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety: clip sigma and cost to avoid division by zero or inf
    # Sigma should be positive, cost should be positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, (f_best - mu) is negative.
    # However, for large negative z, Phi(z) is small, and the term handles it.
    # Standard EI formula assumes we are looking for improvement.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should theoretically be non-negative, but numerical errors might make it slightly negative.
    # We clip to 0 to ensure we don't prefer points that are definitely worse than f_best.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all outputs are finite
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END

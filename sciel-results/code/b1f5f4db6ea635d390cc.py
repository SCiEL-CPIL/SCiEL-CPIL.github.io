"""Cost-aware acquisition for 2D Levy optimization using Probability of Significant Improvement (PSI)."""
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and ensure we return well before the limit.
_DEADLINE = 325.0  # seconds, safety margin of 5s
_start_time = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Significant Improvement (PSI) acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    # Safety check
    if _remaining_time() < 0.5:
        return np.zeros_like(mu)

    n = len(mu)
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probability of Significant Improvement (PSI)
    # P(f(x) < f_best - xi) where xi is a small improvement threshold
    # For minimization, we want the probability that the function value is significantly better than f_best
    # P(f(x) < f_best - xi) = Phi((f_best - xi - mu) / sigma)
    
    # We use a small xi to encourage meaningful improvements
    # As f_best approaches 0, the PSI naturally decays because the threshold f_best - xi becomes very small
    # This provides a principled exploration/exploitation balance without ad-hoc scaling
    
    xi = 0.01  # small improvement threshold in objective units
    
    # Calculate PSI
    # z = (f_best - xi - mu) / sigma
    # PSI = Phi(z)
    
    # Handle numerical stability
    z = (f_best - xi - mu) / sigma
    # Clip z to avoid extreme values that might cause numerical issues
    z = np.clip(z, -10, 10)
    
    psi = norm.cdf(z)
    
    # Cost-aware scoring: divide by cost to penalize expensive evaluations
    # Higher PSI and lower cost lead to higher scores
    scores = psi / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

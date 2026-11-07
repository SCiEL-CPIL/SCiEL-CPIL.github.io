"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
Strategy: Global budget-aware scaling. Instead of point-wise division by cost,
we compute a normalized expected improvement score and subtract a penalty term
proportional to the candidate's cost relative to the batch mean cost.
This shifts the exploration-exploitation balance based on the remaining budget
and the relative expense of each candidate, rather than using local cost ratios.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def _expected_improvement(mu, sigma, f_best):
    """
    Compute Expected Improvement for minimization.
    EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2)
    """
    sigma_safe = np.clip(sigma, 1e-12, None)
    improvement = f_best - mu
    z = improvement / sigma_safe
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI should be non-negative
    return np.clip(ei, 0.0, None)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using global budget-aware scaling.
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe random-like score
    # to allow the loop to finish or fail gracefully without hanging.
    if _is_deadline_exceeded():
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # Compute Expected Improvement
    ei_scores = _expected_improvement(mu, sigma_safe, f_best)
    
    # Normalize EI scores to [0, 1] range for consistent scaling
    ei_max = np.max(ei_scores)
    if ei_max > epsilon:
        ei_normalized = ei_scores / ei_max
    else:
        ei_normalized = np.zeros_like(ei_scores)
    
    # Compute cost z-score: how expensive each candidate is relative to the batch mean
    cost_mean = np.mean(cost_safe)
    cost_std = np.std(cost_safe)
    if cost_std > epsilon:
        cost_z = (cost_safe - cost_mean) / cost_std
    else:
        cost_z = np.zeros_like(cost_safe)
    
    # Penalty term: proportional to cost z-score
    # Higher cost than average -> higher penalty -> lower score
    # Lower cost than average -> lower penalty -> higher score
    # Scale the penalty to be meaningful relative to the EI score range
    penalty_scale = 0.5
    penalty = penalty_scale * cost_z
    
    # Final score: normalized EI minus cost penalty
    # This rewards high expected improvement and low relative cost
    scores = ei_normalized - penalty
    
    # Ensure scores are finite and non-negative (clip to 0 for safety)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, 0.0, None)
    
    return scores
# EVOLVE-BLOCK-END

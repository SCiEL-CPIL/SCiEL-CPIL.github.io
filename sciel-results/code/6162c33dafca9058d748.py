"""Rewritten program using cost-aware Expected Improvement (EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
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
        # Return slightly random scores to break ties, or just zeros
        # Zeros are safe because the evaluator picks argmax. 
        # If all are 0, it picks index 0.
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability: prevent sigma from being 0 or negative
    # A very small epsilon prevents division by zero in z = (f_best - mu) / sigma
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement I(x) = max(f_best - f(x), 0)
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be by definition, but numerical errors can occur)
    ei = np.clip(ei, 0.0, None)
    
    # Cost-awareness:
    # We want to maximize the expected improvement per unit cost.
    # This is a standard heuristic for cost-aware Bayesian optimization.
    # Score = EI / cost
    # This encourages the algorithm to pick points that offer high expected improvement
    # relative to their evaluation cost.
    
    scores = ei / cost_safe
    
    # Final safety: replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END

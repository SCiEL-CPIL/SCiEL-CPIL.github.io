"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 315.0  # safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    Computes the Expected Improvement for each candidate based on the GP
    posterior mean (mu) and standard deviation (sigma), then normalizes by
    the evaluation cost to balance improvement against cost.
    
    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate
                candidates that are more likely to reduce the regret
                per unit cost.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate the z-score: how many standard deviations f_best is above mu
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # For minimization, EI is the expected amount by which f(x) improves f_best
    # EI = (f_best - mu) * P(f(x) < f_best) + sigma * E[f(x) | f(x) < f_best]
    # Using the standard normal CDF (Phi) and PDF (phi):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Cost-aware score: EI divided by cost
    # This encourages both high improvement and low cost
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

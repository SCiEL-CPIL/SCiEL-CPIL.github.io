"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement plus Uncertainty Bonus per unit cost.
    
    This approach models the acquisition as a stochastic shortest path problem where
    we maximize the expected reduction in regret per unit cost, while explicitly
    accounting for the potential (uncertainty) to find a significantly better minimum.
    
    Score = (EI(x) + beta * sigma(x)) / cost(x)
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-12)
        cost_safe = np.maximum(cost, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = (ei + 1.5 * sigma_safe) / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty Bonus: Explicitly value the potential for large improvements
    # This acts as an exploration term in the stochastic shortest path context,
    # representing the "option value" of the candidate.
    beta = 1.5
    uncertainty_bonus = beta * sigma_safe
    
    # Total Value: Expected Improvement + Option Value
    total_value = ei + uncertainty_bonus
    
    # Cost-Aware Score: Value per unit cost
    score = total_value / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: the evaluator kills after 330s wall-clock.
# We track a global start time to ensure we return well before the limit.
_GLOBAL_START = time.time()
_DEADLINE = 330.0 - 10.0  # 10s safety margin

def _time_exceeded():
    return (time.time() - _GLOBAL_START) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for each candidate.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: float, best observed value so far
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of scores, higher = more desirable to evaluate next
    """
    # Check deadline
    if _time_exceeded():
        # Return a simple fallback: prioritize low cost, high (f_best - mu)
        return (f_best - mu) / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (EI should theoretically be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Compute mean cost of the batch for global budget-aware scaling
    mean_cost = np.mean(cost)
    
    # Global budget-aware scaling strategy:
    # Instead of dividing each EI by its local cost, we compute a normalized EI
    # and subtract a penalty proportional to (cost - mean_cost) / mean_cost.
    # This shifts the exploration-exploitation balance based on remaining budget.
    #
    # The idea: candidates with cost below mean get a boost (penalty is negative),
    # candidates with cost above mean get a penalty. This is a global shift rather
    # than a point-wise division.
    #
    # score = EI / mean_cost - lambda * (cost - mean_cost) / mean_cost
    # where lambda controls the strength of the cost penalty.
    
    # A reasonable lambda: we want the penalty to be significant but not dominate.
    # Let's use lambda = 0.5, which means a candidate with cost = 2*mean_cost
    # gets a penalty of 0.5 * (2-1) = 0.5, while EI/mean_cost is typically ~0.1-0.5.
    lambda_penalty = 0.5
    
    # Normalize EI by mean cost to get a scale-invariant improvement rate
    normalized_ei = ei / (mean_cost + 1e-9)
    
    # Cost penalty relative to mean
    cost_penalty = lambda_penalty * (cost - mean_cost) / (mean_cost + 1e-9)
    
    # Final score
    scores = normalized_ei - cost_penalty
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

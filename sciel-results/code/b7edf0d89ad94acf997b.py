import numpy as np
import time
from scipy.stats import norm

# Deadline guard: ensure we finish well before 330s wall-clock
_DEADLINE = time.time() + 290.0  # safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement (EI) normalized by cost.
    
    Strategy:
      - Compute Expected Improvement for minimization:
        EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
      - Score = EI / cost
      - This directly optimizes expected regret reduction per unit cost.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -20, 20)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: expected improvement per unit cost
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

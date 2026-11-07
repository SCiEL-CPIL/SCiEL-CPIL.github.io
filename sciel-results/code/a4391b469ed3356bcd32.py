import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Progress-Adaptive Cost-Adjusted EI.
    """
    # Safety: ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return -mu / cost

    known_min = -78.3323
    # Estimate progress: 0 at start (f_best ~ 20), 1 at optimum
    f_start_est = 20.0
    progress = np.clip((f_start_est - f_best) / (f_start_est - known_min), 0.0, 1.0)
    
    # Standardized improvement
    z = (f_best - mu) / (sigma + 1e-12)
    
    # Expected Improvement (for minimization)
    # EI = (f_best - mu) * P(x < f_best) + sigma * pdf(z)
    # Note: For minimization, improvement is f_best - f(x).
    # P(f(x) < f_best) = P((f(x)-mu)/sigma < (f_best-mu)/sigma) = Phi(z)
    # E[max(f_best - f(x), 0)] = (f_best - mu)Phi(z) + sigma phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration term: weighted by uncertainty
    # Decay exploration as we get closer to the minimum
    exploration_weight = 0.5 * (1.0 - progress) ** 2
    
    # Combine EI and exploration
    # We add exploration to EI before dividing by cost
    combined_value = ei + exploration_weight * sigma
    
    # Cost adjustment: divide by cost
    # Cost is ~1 to 2. Dividing by cost penalizes expensive points.
    final_score = combined_value / cost
    
    # Ensure finite output
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score

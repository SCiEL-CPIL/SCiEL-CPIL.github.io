# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level constants for time management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill

# Fixed hyperparameters for the acquisition function
# Based on the directive: Cost-Adjusted PoI + small GP uncertainty bonus
# beta controls the exploration-exploitation trade-off
# Using a small beta to prioritize exploitation (PoI) but allow some exploration
BETA_EXPLORATION = 0.1

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Probability of Improvement (PoI).
    
    Parameters
    ----------
    mu     : (N,) array of GP mean predictions
    sigma  : (N,) array of GP standard deviations
    f_best : float, the best (lowest) observed value so far
    cost   : (N,) array of evaluation costs for each candidate

    Returns
    -------
    scores : (N,) array of finite scores, higher = evaluate next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: return random finite scores to avoid invalidity
        return np.random.rand(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero in z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    f_best = float(f_best)
    
    # Calculate Probability of Improvement: P(f(x) < f_best)
    # Assuming normal distribution of f(x) with mean mu and std sigma
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Cost-Adjusted PoI: Prioritize candidates with high PoI and low cost
    # Score component 1: PoI / cost
    score_poi = poi / cost_safe
    
    # Score component 2: Exploration bonus
    # Uncertainty per unit cost
    score_expl = BETA_EXPLORATION * (sigma_safe / cost_safe)
    
    # Total score
    scores = score_poi + score_expl
    
    # Ensure all scores are finite
    # If any are not finite, replace them with a very low value
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e9)
        
    return scores
# EVOLVE-BLOCK-END

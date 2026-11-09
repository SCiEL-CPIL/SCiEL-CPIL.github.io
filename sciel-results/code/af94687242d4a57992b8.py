# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level constants for time management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill

# Hyperparameters for the acquisition function
# Structural Change: Switch from PoI to Expected Improvement (EI)
# with a multiplicative cost discount and additive cost-adjusted exploration.
BETA_EXPLORATION = 0.3   # Weight for the exploration term (sigma/cost)
DELTA_COST = 0.5         # Exponent for the cost discount on EI: (mean_cost / cost)^delta

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Expected Improvement (EI)
    with cost discounting and exploration.
    
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
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # Assuming f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) is norm.cdf(z), phi(z) is norm.pdf(z)
    # Using scipy.stats.norm for numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Calculate Cost Discount
    # We prefer cheaper points.
    # discount = (mean_cost / cost) ** DELTA_COST
    # If cost < mean_cost, discount > 1 (boost)
    # If cost > mean_cost, discount < 1 (penalty)
    mean_cost = np.mean(cost_safe)
    if mean_cost <= 0:
        mean_cost = 1.0
    cost_discount = (mean_cost / cost_safe) ** DELTA_COST
    
    # Score Component 1: Cost-Adjusted Expected Improvement
    score_ei = ei * cost_discount
    
    # Score Component 2: Exploration Bonus
    # Uncertainty per unit cost
    score_expl = BETA_EXPLORATION * (sigma_safe / cost_safe)
    
    # Total Score
    scores = score_ei + score_expl
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e9)
        
    return scores
# EVOLVE-BLOCK-END

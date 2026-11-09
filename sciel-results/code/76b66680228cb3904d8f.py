import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a power-law cost penalty and exploration bonus.
    
    Args:
        mu: GP mean predictions (2000,)
        sigma: GP standard deviation predictions (2000,)
        f_best: Best function value observed so far (scalar)
        cost: Cost of evaluating each candidate (2000,)
        
    Returns:
        Acquisition scores (2000,), higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a reasonable fallback: pure EI/cost without extra computation
        cost_safe = np.maximum(cost, 1e-9)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage exploring high-uncertainty regions
    # when EI is low (i.e., when we're not clearly improving)
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma_safe
    
    # Combined acquisition before cost adjustment
    acq = ei + exploration_bonus
    
    # Cost-aware: use power-law penalty (cost^0.7) instead of linear
    # This more smoothly trades off improvement vs cost
    acq /= np.power(cost_safe, 0.7)
    
    # Handle any numerical issues (NaN, Inf)
    acq = np.nan_to_num(acq, nan=0.0, posinf=0.0, neginf=0.0)
    
    return acq

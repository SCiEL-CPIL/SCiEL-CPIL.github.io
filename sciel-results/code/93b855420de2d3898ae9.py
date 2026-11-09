import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and a small exploration bonus based on sigma.
    
    Args:
        mu: GP mean predictions (2000,)
        sigma: GP standard deviation predictions (2000,)
        f_best: Best function value observed so far (scalar)
        cost: Cost of evaluating each candidate (2000,)
        
    Returns:
        Acquisition scores (2000,), higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Small exploration bonus to prevent getting stuck in local minima
    # or regions of high confidence but poor value.
    # Based on knowledge, a small bonus (0.05) performed well.
    exploration_bonus = 0.05 * sigma_safe
    
    # Combined acquisition
    acq = ei + exploration_bonus
    
    # Cost-aware: divide by cost
    # Cost is always >= 1, so this is safe
    acq /= cost
    
    # Handle any numerical issues (NaN, Inf)
    acq = np.nan_to_num(acq, nan=0.0, posinf=0.0, neginf=0.0)
    
    return acq

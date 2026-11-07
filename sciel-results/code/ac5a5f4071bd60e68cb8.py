import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Quantile of Improvement (QI) acquisition function with exploration bonus.
    
    QI focuses on high-probability, high-reward regions by maximizing the
    conditional expected value of improvement given that improvement occurs.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Using (f_best - mu) because we are minimizing
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Calculate Probability of Improvement (PI)
    pi = norm.cdf(z)
    pi_safe = np.maximum(pi, 1e-12)
    
    # Quantile of Improvement (QI)
    # QI = EI / PI, the conditional expected improvement given improvement occurs
    # This focuses on regions where improvement is both likely and large
    qi = ei / pi_safe
    
    # Clip QI to avoid extreme values
    qi = np.clip(qi, 0.0, 1e6)
    
    # Add exploration bonus proportional to sigma/cost
    # This encourages sampling in high-uncertainty regions
    # The bonus is scaled to be meaningful relative to QI
    exploration_bonus = 0.3 * sigma_safe / cost
    
    # Cost-aware score: Maximize QI per unit cost + exploration bonus
    # We divide QI by cost to prefer cheaper points if QI is comparable
    score = qi / cost + exploration_bonus
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score

import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    This function calculates the Probability of Improvement for each candidate point
    and normalizes it by the evaluation cost to maximize efficiency.
    PI is often more robust in multimodal landscapes than EI because it focuses
    on the likelihood of finding any improvement, rather than the expected magnitude.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Deadline guard
    t_start = time.time()
    deadline = 300.0
    
    # Ensure sigma is positive to avoid division by zero
    # Add a small epsilon to prevent numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, PI > 0.5
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is the CDF of the standard normal distribution evaluated at z
    pi_scores = norm.cdf(z)
    
    # Cost-aware score: Maximize PI per unit cost
    # We divide by cost to prioritize cheaper evaluations with high probability of improvement
    # Using a power of cost (e.g., cost**1) is standard.
    score = pi_scores / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > deadline - 5.0:
        # Safety margin: return current result
        return score
    
    return score

import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes a weighted sum of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    for minimization.
    
    Args:
        mu: Mean of the Gaussian Process posterior, shape (N,)
        sigma: Standard deviation of the GP posterior, shape (N,)
        f_best: Best observed value so far.
        
    Returns:
        Acquisition scores, shape (N,). Higher is better.
    """
    # Safety margin check (though this function is fast)
    # This is a placeholder for the "deadline guard" requirement, 
    # but since we cannot abort the caller, we just ensure we don't do anything infinite.
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB for minimization
    # We want to maximize the score. 
    # Low mu is good. High sigma is good (uncertainty).
    # UCB_score = -mu + beta * sigma
    beta = 2.0
    ucb_score = -mu + beta * sigma_safe
    
    # Weighted sum
    # Based on the hint, a direct weighted sum works well.
    # EI is already scaled by the gap to f_best. UCB is in objective units.
    # They are on similar scales if f_best is not extremely far from mu.
    # Let's use equal weights or slightly favor EI.
    # The hint said "Direct weighted sum ... better balancing".
    # Let's try 1.0 * EI + 1.0 * UCB_score.
    
    scores = ei + ucb_score
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

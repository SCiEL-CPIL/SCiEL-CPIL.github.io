import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for safety margin
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) and
    Lower Confidence Bound (LCB) for minimization.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Safety check: if we are getting close to the 330s limit, return a simple result
    # This function is vectorized and fast, but we include the guard as per directive
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return a simple exploration-focused score as a fallback
        return sigma
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # 1. Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # We want to maximize the acquisition score, so we use -LCB = beta * sigma - mu
    # This encourages exploration in high-uncertainty regions (high sigma)
    # and exploitation in low-prediction regions (low mu).
    # Based on findings, beta=3.0 provides better global coverage in high dimensions.
    beta = 3.0
    lcb_score = beta * sigma_safe - mu
    
    # 3. Hybrid combination
    # We blend EI and LCB to balance local exploitation (EI) and global exploration (LCB).
    # A 50/50 mix is a reasonable starting point, but we can weight LCB slightly higher
    # to ensure sufficient exploration in the 6D space with only 30 evaluations.
    # Let's use a weighted sum: 0.4 * EI + 0.6 * LCB_score
    # This structure allows the LCB term to drive exploration when EI is small,
    # while EI provides a strong signal when a promising improvement is detected.
    
    scores = 0.4 * ei + 0.6 * lcb_score
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

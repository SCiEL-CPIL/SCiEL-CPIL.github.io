import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
        
    Returns:
    array-like, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Deadline guard
    current_time = time.time()
    elapsed = current_time - _start_time
    if elapsed > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fallback: Return scores based purely on exploitation (lowest mu) 
        # or just random to avoid complex computation? 
        # Better: Return EI without cost adjustment or just -mu to be fast?
        # The most important thing is to return finite values.
        # Let's return a simple exploitation score: -mu / cost
        # This is very fast.
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / np.maximum(cost, 1e-8)

    # Standard Cost-Adjusted Expected Improvement
    # Add a small exploration term xi to avoid local optima
    # xi is chosen to be a small fraction of the current best value or a fixed small number
    # Since f_best decreases, a fixed small xi like 0.01 is often safe for Ackley.
    # Let's use xi = 0.01 * max(1, f_best) to scale slightly with the problem scale.
    xi = 0.01 * max(1.0, f_best)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for EI
    # z = (f_best - mu - xi) / sigma
    numerator = f_best - mu - xi
    z = numerator / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu - xi < 0, EI should be 0. 
    # The formula (A * Phi(z) + sigma * phi(z)) naturally handles this if z is negative enough?
    # Actually, if A < 0, z < 0. Phi(z) is small, phi(z) is positive.
    # The term A*Phi(z) is negative. sigma*phi(z) is positive.
    # For large negative z, EI -> 0.
    # To be safe and strictly non-negative:
    ei = np.where(numerator > 0, 
                  numerator * norm.cdf(z) + sigma_safe * norm.pdf(z), 
                  0.0)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-adjusted score
    scores = ei / cost_safe
    
    # Handle any numerical issues (NaN/Inf)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

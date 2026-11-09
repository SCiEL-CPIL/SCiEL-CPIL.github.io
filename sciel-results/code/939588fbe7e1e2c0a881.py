import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using exponential cost suppression.
    
    Score = EI * exp(-k * cost)
    
    Args:
        mu: array of GP means, shape (N,)
        sigma: array of GP standard deviations, shape (N,)
        f_best: scalar, best function value observed so far
        cost: array of costs, shape (N,)
        
    Returns:
        scores: array of acquisition values, shape (N,), higher is better
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margin check, though this function is very fast
    if time.time() - start_time > 290:
        return np.zeros_like(mu)

    # Prevent division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Clip negative EI to 0
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Multiplicative Exponential Cost Gate
    # Instead of dividing by cost, we multiply by exp(-k * cost)
    # This creates a sharp penalty for high-cost points.
    # k is calibrated to the cost range [1, 2].
    # If k=1, cost=2 is penalized by exp(-2) ~ 0.135 relative to cost=1 (exp(-1) ~ 0.368).
    # Ratio is exp(1) ~ 2.7.
    # If k=2, ratio is exp(2) ~ 7.4.
    # Let's use k=1.5 for a strong but not prohibitive penalty.
    k = 1.5
    cost_gate = np.exp(-k * cost)
    
    scores = ei * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final time check
    if time.time() - start_time > 295:
         # If we are extremely close to timeout, ensure we return something valid quickly
         # The calculation above is vectorized and fast, so this is unlikely to trigger
         pass
         
    return scores

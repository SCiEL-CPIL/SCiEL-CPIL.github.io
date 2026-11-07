import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost exponent
    # Early on (high f_best), we are less sensitive to cost (alpha close to 1).
    # Late on (low f_best), we are more sensitive to cost (alpha close to 2).
    # f_best starts high (e.g., 50) and goes to 0.
    # We want alpha to increase as f_best decreases.
    # alpha = 1.0 + 1.0 / (1.0 + f_best)
    # If f_best = 10, alpha = 1.09
    # If f_best = 1, alpha = 1.5
    # If f_best = 0, alpha = 2.0
    alpha = 1.0 + 1.0 / (1.0 + np.maximum(f_best, 0.0))
    
    # Cost-aware score
    scores = ei / np.power(cost_safe, alpha)
    
    # Add a small exploration term to avoid zero scores if EI is 0
    # This helps escape local minima in the acquisition landscape
    exploration = 0.1 * sigma_safe / np.power(cost_safe, alpha)
    scores += exploration
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores

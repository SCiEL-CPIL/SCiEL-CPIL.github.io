import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    Args:
        mu: Mean of the GP prediction, shape (2000,)
        sigma: Standard deviation of the GP prediction, shape (2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,), higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: PI per unit cost
    # We want to maximize this
    scores = pi / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores

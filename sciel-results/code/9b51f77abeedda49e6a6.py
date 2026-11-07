# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    This approach selects points that are most likely to yield a better value
    than the current best, penalized by the evaluation cost.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    Score = PI(x) / cost(x)
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score for Probability of Improvement
    # We want P(f(x) < f_best)
    # Since f(x) ~ N(mu, sigma^2), 
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Calculate PI
    pi = norm.cdf(z)
    
    # Cost-aware score: Higher PI per unit cost is better
    # cost is always >= 1 in this problem (1 + mean(x)), so no division by zero risk
    score = pi / np.maximum(cost, 1e-10)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

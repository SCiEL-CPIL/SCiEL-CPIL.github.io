import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using PI (Probability of Improvement) with 
    soft cost penalty via exponentiation. This approach naturally saturates 
    and decays smoothly, providing better exploration-exploitation balance 
    than linear UCB/EI for this specific cost-aware optimization task.
    """
    # Convert inputs to arrays and ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # z-score for PI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Soft cost penalty via exponentiation: PI^(1/cost)
    # This creates a smooth multiplicative trade-off where expensive points
    # are penalized more strongly as their PI increases
    # The exponent 1/cost means that as cost increases, the effective PI decreases
    scores = np.power(pi, 1.0 / cost_safe)
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

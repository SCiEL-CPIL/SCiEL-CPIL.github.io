import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) 
    with an uncertainty bonus and soft cost penalty.
    
    Score = (EI + beta * sigma) / (cost ** alpha)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (Gaussian)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Parameters
    # Beta: Weight of the uncertainty bonus (exploration)
    # Alpha: Exponent for cost penalty (softness)
    beta = 1.5
    alpha = 0.8
    
    # Uncertainty Bonus
    # This encourages exploring regions with high predictive variance
    ub = beta * sigma_safe
    
    # Numerator: EI + Uncertainty Bonus
    numerator = ei + ub
    
    # Denominator: Soft Cost Penalty
    # Using cost**alpha with alpha < 1 penalizes high cost less than linear division
    denominator = np.power(cost_safe, alpha)
    
    scores = numerator / denominator
    
    # Ensure output is finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

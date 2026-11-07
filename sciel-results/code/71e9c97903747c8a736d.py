import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the Probability of Improvement (PI)
    # This is the probability that f(x) < f_best
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Ensure cost is at least 1 to avoid division by very small numbers
    # The problem states cost = 1 + mean(x_scaled), so cost >= 1
    cost_safe = np.maximum(cost, 1.0)
    
    # Dynamic uncertainty budgeting:
    # beta decays as total spent cost approaches the budget limit (30)
    # to transition from exploration to exploitation
    
    # Estimate remaining budget from the cost array
    # cost is the per-evaluation cost, but we need total spent cost
    # Since we don't have total spent cost directly, we use a proxy:
    # The mean cost gives us an indication of how expensive evaluations have been
    # We assume the budget is 30 and use the mean cost to estimate progress
    
    # A simpler approach: use a fixed exploration-exploitation trade-off
    # that adapts based on the uncertainty in the model
    # When sigma is high (early on), explore more; when sigma is low, exploit
    
    # Calculate a dynamic beta that encourages exploration when uncertainty is high
    # and exploitation when uncertainty is low
    # beta = base_beta * (1 + uncertainty_factor)
    # where uncertainty_factor is based on the mean sigma relative to the scale
    
    # Normalize sigma to get a sense of relative uncertainty
    sigma_mean = np.mean(sigma_safe)
    # Use a sigmoid-like function to map uncertainty to exploration level
    # High uncertainty -> high beta -> more exploration
    # Low uncertainty -> low beta -> more exploitation
    
    # The exploration term is beta * sigma (UCB-like)
    # The exploitation term is the probability of improvement
    # We combine them as: score = pi + beta * sigma / cost
    
    # Choose beta based on the overall uncertainty level
    # When mean sigma is large, we want more exploration
    # When mean sigma is small, we want more exploitation
    
    # A simple heuristic: beta = 1.0 when sigma_mean is above a threshold,
    # and decreases as sigma_mean decreases
    # Let's use: beta = max(0.1, min(2.0, sigma_mean * 2))
    
    beta = np.clip(sigma_mean * 2.0, 0.1, 2.0)
    
    # Cost-aware score: combine PI and exploration term
    # score = (pi + beta * sigma_safe) / cost_safe
    exploration_term = beta * sigma_safe
    score = (pi + exploration_term) / cost_safe
    
    # Replace any non-finite values with 0 to ensure validity
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # We want to maximize the potential improvement per unit cost.
    # Potential improvement is estimated by f_best - (mu - beta * sigma)
    # which is f_best - mu + beta * sigma.
    
    # Beta controls the exploration/exploitation trade-off.
    # For Hartmann 3D with Matern 5/2 and LS 0.2, beta ~ 2.0-3.0 is typical.
    beta = 2.5
    
    # Calculate the score
    # Ensure sigma is positive to avoid issues, though GP sigma should be > 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # The numerator is the estimated improvement over f_best using the lower bound
    improvement = f_best - mu + beta * sigma_safe
    
    # Divide by cost to penalize expensive evaluations
    # cost is shape (2000,)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = improvement / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

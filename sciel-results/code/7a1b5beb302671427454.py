import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Standardized Cost-Effective UCB.
    
    This approach uses an Upper Confidence Bound (UCB) for minimization, 
    standardized to be scale-invariant, and divided by the evaluation cost.
    This explicitly optimizes for the expected reduction in objective value 
    per unit of cost, while maintaining a balance between exploration 
    (high sigma) and exploitation (low mu).
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs to ensure numerical stability
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calculate statistics for standardization
    # We use a small epsilon to prevent division by zero
    eps = 1e-8
    
    mu_mean = np.mean(mu)
    mu_std = np.std(mu) + eps
    
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + eps
    
    # Standardize mu and sigma
    # For minimization, we want low mu and high sigma.
    # The UCB score for minimization (to be maximized) is: -mu + beta * sigma
    # In standardized form: -z_mu + beta * z_sigma
    z_mu = (mu - mu_mean) / mu_std
    z_sigma = (sigma - sigma_mean) / sigma_std
    
    # Beta controls the exploration/exploitation trade-off.
    # A value of 2.0 provides a strong exploration incentive.
    beta = 2.0
    
    # Calculate the raw UCB score
    ucb_score = -z_mu + beta * z_sigma
    
    # Apply cost penalty (divide by cost)
    # This makes the acquisition function "cost-effective".
    # Points with lower cost (closer to 0 in the domain) are favored,
    # which aligns with the Powell function's minimum at the origin.
    scores = ucb_score / cost
    
    # Deadline guard
    if time.time() - t_start > 2.0:
        # Fallback to pure exploration if computation is unexpectedly slow
        return z_sigma / cost
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

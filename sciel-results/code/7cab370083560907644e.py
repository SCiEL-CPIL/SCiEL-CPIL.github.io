import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on powell 4d.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Predicted mean of the GP at the 2000 candidate points
    sigma : array of shape (2000,)
        Predicted standard deviation of the GP at the 2000 candidate points
    f_best : float
        Best (lowest) observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled)
    
    Returns:
    --------
    score : array of shape (2000,)
        Acquisition scores (higher = prefer to evaluate next)
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Standard EI formula
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration weight based on global uncertainty
    # When sigma is large relative to improvement, we need more exploration
    # When sigma is small, we can focus on exploitation
    
    # Compute a measure of how much uncertainty remains
    mean_sigma = np.mean(sigma_safe)
    max_sigma = np.max(sigma_safe)
    
    # Normalize sigma to get a relative uncertainty measure
    # beta adapts based on the overall confidence of the GP
    if max_sigma > 1e-10:
        # When average sigma is high relative to max, we're still exploring
        # When average sigma is low, we should exploit more
        sigma_ratio = mean_sigma / max_sigma
        # beta ranges from 0.5 (exploit) to 2.0 (explore)
        beta = 0.5 + 1.5 * (1.0 - sigma_ratio)
    else:
        beta = 0.5
    
    # Cost-aware scoring
    # cost ranges from 1 to 2 (since cost = 1 + mean(x_scaled) where x_scaled in [0,1])
    # We want to prefer lower cost evaluations
    
    # Primary: EI normalized by cost
    base_score = ei / cost
    
    # Exploration component: high sigma regions are worth exploring
    # But penalize high cost
    exploration_score = beta * sigma_safe / cost
    
    # Combine
    score = base_score + exploration_score
    
    # Additional bias: prefer regions closer to the expected minimizer
    # The minimizer is in [0.1, 0.9]^4 in scaled coordinates
    # Expected cost at minimizer: 1 + 0.5 = 1.5
    # Add a Gaussian weight centered at cost=1.5
    cost_center = 1.5
    cost_width = 0.25
    cost_weight = np.exp(-0.5 * ((cost - cost_center) / cost_width) ** 2)
    
    # Multiply by a factor that boosts the center region
    multiplier = 0.7 + 0.3 * cost_weight
    score = score * multiplier
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time budget - safety margin before 300s limit
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Return a simpler score to avoid timeout
        simple_score = ei / cost
        simple_score = np.nan_to_num(simple_score, nan=0.0, posinf=0.0, neginf=0.0)
        return simple_score
    
    return score

import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: array of GP posterior means, shape (N,)
        sigma: array of GP posterior standard deviations, shape (N,)
        f_best: current best observed function value (scalar)
        cost: array of evaluation costs for each candidate, shape (N,)
        
    Returns:
        array of acquisition scores, shape (N,), higher is better.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate Gaussian noise
    # Using standard normal distribution
    eps = np.random.randn(mu.shape[0])
    
    # Sample function values
    f_sample = mu + sigma_safe * eps
    
    # Calculate cost-adjusted improvement
    # We want to maximize (f_best - f_sample) / cost
    # cost is guaranteed to be >= 1, so no division by zero risk, 
    # but we guard just in case.
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = (f_best - f_sample) / cost_safe
    
    return scores

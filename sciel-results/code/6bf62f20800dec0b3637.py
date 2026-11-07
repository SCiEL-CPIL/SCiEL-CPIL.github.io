import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed value (not directly used in TS sampling, but part of signature).
    cost : array-like, shape (2000,)
        Cost of evaluating each point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sample a single objective value from the Gaussian posterior for each candidate
    # y ~ N(mu, sigma)
    # Using standard normal noise
    eps = np.random.normal(size=mu.shape)
    y = mu + sigma * eps
    
    # Cost-aware score:
    # We want to minimize the objective y.
    # We want to minimize the cost.
    # Score should be high for good (low y) and low cost.
    # score = -y / cost
    # This prioritizes points with low predicted value and low cost.
    
    cost_safe = np.maximum(cost, 1e-9)
    scores = -y / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

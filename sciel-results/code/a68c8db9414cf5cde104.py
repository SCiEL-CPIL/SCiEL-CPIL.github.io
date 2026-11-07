import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Parameters:
    -----------
    mu : numpy.ndarray
        Mean of the GP predictive distribution, shape (N,).
    sigma : numpy.ndarray
        Standard deviation of the GP predictive distribution, shape (N,).
    f_best : float
        The lowest observed function value so far.
    cost : numpy.ndarray
        Cost of evaluating each candidate, shape (N,).
    
    Returns:
    --------
    scores : numpy.ndarray
        Acquisition scores, shape (N,). Higher is better.
    """
    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    # The problem states cost = 1 + mean(x_scaled), so cost >= 1.
    # But just in case, clip to a small positive number.
    cost_safe = np.maximum(cost, 1e-6)
    
    # Thompson Sampling:
    # Sample a single realization of the latent function for all candidates.
    # epsilon ~ N(0, 1)
    eps = np.random.standard_normal(mu.shape)
    
    # Sampled function values
    # Since we are minimizing, lower values are better.
    # The acquisition score should be higher for better candidates.
    # So we use negative sampled values.
    f_sampled = mu + sigma * eps
    score_base = -f_sampled
    
    # Cost-awareness
    # The knowledge base suggests cost^0.7 is better than cost^1.
    # This reduces the penalty for high-cost points, allowing exploration in 
    # potentially valuable but expensive regions.
    alpha = 0.7
    cost_penalty = cost_safe ** alpha
    
    scores = score_base / cost_penalty
    
    # Ensure finiteness
    # If any values are not finite, replace with a very low score (worst case)
    # to avoid invalidating the result.
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e10)
        
    return scores

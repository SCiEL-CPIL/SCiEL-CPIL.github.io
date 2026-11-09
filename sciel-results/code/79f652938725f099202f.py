import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: Array of shape (2000,) with GP mean predictions.
        sigma: Array of shape (2000,) with GP standard deviation predictions.
        f_best: Scalar, the best function value observed so far.
        cost: Array of shape (2000,) with the cost of evaluating each candidate.
        
    Returns:
        Array of shape (2000,) with scores. Higher scores indicate better candidates.
    """
    # Ensure inputs are numpy arrays for efficient computation
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost (though cost >= 1)
    cost = np.maximum(cost, 1e-9)
    
    # Avoid negative or zero sigma for sampling stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from the posterior for each candidate
    # y_i ~ N(mu_i, sigma_i^2)
    # Using standard normal and scaling is efficient
    z = np.random.normal(0.0, 1.0, size=mu.shape)
    y_samples = mu + sigma * z
    
    # Cost-aware score: We want to minimize y_samples / cost.
    # The acquisition function should return higher values for better candidates.
    # So we return - (y_samples / cost).
    scores = - (y_samples / cost)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

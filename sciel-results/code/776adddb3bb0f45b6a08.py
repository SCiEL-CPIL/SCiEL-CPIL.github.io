import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (N,) with the cost of evaluating each point.
        
    Returns:
        Array of shape (N,) with acquisition scores. Higher is better.
    """
    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Prevent division by zero or extremely small sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB parameter: controls the balance between exploitation (low mu)
    # and exploration (high sigma).
    # Higher beta encourages exploration.
    beta = 2.0
    
    # Calculate the acquisition score
    # For minimization, we want to maximize:
    # (Expected Improvement Potential) / Cost
    # We use the UCB form: f_best - mu + beta * sigma
    # This encourages points with low mu (good mean) and high sigma (uncertain).
    # Dividing by cost prioritizes cheaper evaluations.
    
    numerator = (f_best - mu) + beta * sigma
    score = numerator / cost
    
    # Handle any numerical issues (inf, nan)
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

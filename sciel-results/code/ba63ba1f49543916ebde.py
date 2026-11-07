import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive UCB.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    # Adaptive Beta for UCB
    # We want to explore more when f_best is high (early in optimization)
    # and exploit more when f_best is low (late in optimization).
    # For Griewank, f_best is in [0, 1].
    # Map f_best to a beta range, e.g., [0.5, 2.0].
    # beta = 0.5 + 1.5 * f_best
    
    # Clip f_best to [0, 1] to be safe, though it should be within this range
    f_best_clipped = np.clip(f_best, 0.0, 1.0)
    beta = 0.5 + 1.5 * f_best_clipped
    
    # UCB Score for Minimization:
    # We want to minimize f(x).
    # The lower confidence bound is L(x) = mu(x) - beta * sigma(x).
    # We want to select x that minimizes L(x).
    # Equivalently, we maximize -L(x) = beta * sigma(x) - mu(x).
    # We also want to account for cost, so we divide by cost.
    
    score = (beta * sigma - mu) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

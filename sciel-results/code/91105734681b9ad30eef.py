import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean predictions (2000,)
        sigma: Standard deviation predictions (2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each candidate (2000,)
        
    Returns:
        scores: Acquisition scores (2000,), higher is better.
    """
    # UCB for minimization: mu - beta * sigma
    # We want to maximize the potential improvement relative to cost.
    # Potential improvement proxy: beta * sigma - mu
    # Normalized by cost.
    
    beta = 2.5
    
    # Calculate raw scores
    # Using np.where to handle potential division by zero or invalid values
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = (beta * sigma - mu) / cost
        
    # Replace NaN and Inf with finite values
    # If a score is Inf, it means it's extremely good (or cost is 0, which isn't possible).
    # We clip to a large finite number to satisfy "finite scores" requirement.
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Clip to a reasonable range to prevent numerical overflow in downstream processes
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores

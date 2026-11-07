import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Linear Lower Confidence Bound (LCB) divided by cost.
    
    Uses LCB to balance exploration and exploitation. For minimization, the
    optimistic value is the lower bound mu - beta * sigma. We maximize the
    negative of this bound, normalized by cost, to prefer points that are
    likely to be good and cheap.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior in objective units.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    # Exploration parameter
    beta = 1.5
    
    # Ensure cost is safe for division
    cost_safe = np.maximum(cost, 1e-8)
    
    # LCB for minimization: mu - beta * sigma
    # We want to MINIMIZE this value.
    # To use a "maximize" acquisition function, we take the negative.
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    score = (beta * sigma - mu) / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

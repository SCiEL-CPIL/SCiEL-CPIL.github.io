import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    Args:
        mu: Array of posterior means at candidate points (shape (2000,)).
        sigma: Array of posterior standard deviations at candidate points (shape (2000,)).
        f_best: The lowest observed value so far.
        
    Returns:
        Array of scores (shape (2000,)). Higher score indicates higher priority.
    """
    # LCB for minimization: a(x) = mu(x) - beta * sigma(x)
    # We want to maximize the score, so we return -a(x)
    # Score(x) = -mu(x) + beta * sigma(x)
    
    # Beta = 2.5 is a robust choice for 2D problems with small budgets.
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

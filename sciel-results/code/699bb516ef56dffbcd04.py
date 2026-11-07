import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    For minimization, the LCB is defined as:
    LCB(x) = mu(x) - beta * sigma(x)
    
    We want to find the x that minimizes LCB(x).
    To fit the "higher score is better" interface, we compute:
    Score(x) = - LCB(x) / cost(x) = (beta * sigma(x) - mu(x)) / cost(x)
    
    The division by cost(x) penalizes evaluations that are more expensive.
    """
    # Exploration parameter
    beta = 2.0
    
    # Ensure sigma is non-negative to avoid issues, though it should be by definition
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the cost-adjusted LCB score
    # Higher score indicates a more promising point to evaluate next
    scores = (beta * sigma - mu) / cost
    
    return scores

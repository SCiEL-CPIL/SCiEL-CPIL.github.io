import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    For minimization, we prefer points with low predicted mean and high uncertainty.
    Score = (beta * sigma - mu) / cost
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta encourages more exploration.
    beta = 3.0
    
    # Calculate the Lower Confidence Bound (LCB) for minimization: mu - beta * sigma
    # We want to MINIMIZE this value.
    # Therefore, the acquisition score (to be MAXIMIZED) is the negative of it, adjusted for cost.
    # Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    numerator = beta * sigma - mu
    denominator = cost + 1e-9
    
    scores = numerator / denominator
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores

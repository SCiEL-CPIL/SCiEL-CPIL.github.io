import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    UCB = -mu + beta * sigma
    
    This function addresses the weakness of Expected Improvement (EI) in 
    multimodal landscapes where EI's exploration bonus vanishes for points 
    with poor predicted means. UCB maintains a linear exploration bonus 
    proportional to the standard deviation, encouraging the search to 
    explore high-uncertainty regions even if their current mean is not 
    competitive, thus helping to escape local minima.
    """
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # UCB parameter: controls the balance between exploration and exploitation
    # A higher beta encourages more exploration.
    beta = 2.0
    
    # Compute UCB score
    # We want to minimize f(x), so we maximize -mu + beta * sigma
    ucb_scores = -mu + beta * sigma
    
    return ucb_scores

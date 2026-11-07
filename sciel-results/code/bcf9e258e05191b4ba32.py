import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    For minimization, we want to find points with low expected value (mu)
    and high uncertainty (sigma). The score is defined as:
    Score = -mu + beta * sigma
    
    A higher score indicates a more promising candidate.
    """
    # Beta controls the exploration-exploitation trade-off.
    # A value of 2.0 is a standard default for balancing the two.
    beta = 2.0
    
    # Calculate LCB score
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

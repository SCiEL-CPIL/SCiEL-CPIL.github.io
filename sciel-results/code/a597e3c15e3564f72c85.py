import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    For minimization, the Lower Confidence Bound is:
    LCB(x) = mu(x) - kappa * sigma(x)
    
    We want to minimize LCB, which is equivalent to maximizing:
    Score(x) = -(mu(x) - kappa * sigma(x)) / cost(x)
             = (kappa * sigma(x) - mu(x)) / cost(x)
    
    This encourages exploration (high sigma) and exploitation (low mu),
    penalized by evaluation cost.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Exploration parameter
    kappa = 2.0
    
    # Calculate the score: higher is better
    # (kappa * sigma - mu) represents the "optimism" or lower bound potential
    # Dividing by cost penalizes expensive evaluations
    scores = (kappa * sigma - mu) / cost
    
    # Handle any numerical issues (NaNs, infs)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

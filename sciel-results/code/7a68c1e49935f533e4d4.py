import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Upper Confidence Bound (UCB) to balance exploration and exploitation.
    
    Args:
        mu: Mean of the Gaussian Process posterior (shape: N,)
        sigma: Standard deviation of the Gaussian Process posterior (shape: N,)
        f_best: Best observed function value so far (scalar)
        
    Returns:
        Scores for each candidate (shape: N,), where higher scores indicate
        more promising candidates for evaluation.
    """
    # Aggressive exploration coefficient, based on measured performance improvements
    # for multimodal functions like Weierstrass in low-budget scenarios.
    beta = 2.5
    
    # UCB for minimization: we want to maximize -mu + beta * sigma
    # This favors points with low predicted mean (exploitation) 
    # and high predicted uncertainty (exploration).
    scores = -mu + beta * sigma
    
    return scores

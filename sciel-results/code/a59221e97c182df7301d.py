import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a UCB-like strategy: (f_best - mu + kappa * sigma) / cost
    """
    kappa = 1.0
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate the score
    # We are minimizing, so we want low mu.
    # f_best - mu represents the potential gain if the mean is correct.
    # kappa * sigma represents the uncertainty bonus.
    score = (f_best - mu + kappa * sigma) / cost
    
    # Handle any non-finite values
    score = np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return score

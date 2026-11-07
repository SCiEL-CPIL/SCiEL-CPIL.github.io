import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses Upper Confidence Bound (UCB) strategy.
    
    For minimization, we want to find points that could potentially have
    very low values. The Lower Confidence Bound (LCB) is mu - kappa * sigma.
    We want to MINIMIZE this bound.
    Since the evaluator picks the candidate with the HIGHEST score,
    we return the negative of the LCB: -mu + kappa * sigma.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (N,).
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (N,).
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (N,). Higher score is better.
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter. 
    # A higher value encourages more exploration.
    # Given the short horizon (25 optimization steps) and multimodal objective,
    # a relatively high kappa is appropriate to escape local minima.
    kappa = 2.5
    
    # Score = -mu + kappa * sigma
    # This is equivalent to minimizing (mu - kappa * sigma)
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

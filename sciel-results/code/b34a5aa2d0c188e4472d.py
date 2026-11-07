import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Parameters:
    mu: array of predicted means (shape 2000,)
    sigma: array of predicted standard deviations (shape 2000,)
    f_best: current best observed function value (scalar)
    
    Returns:
    scores: array of acquisition values (shape 2000,), higher is better.
    """
    # Adaptive beta: 
    # In the early stages, the GP is uncertain (high std of mu), so we want to explore more.
    # As the GP converges (low std of mu), we reduce exploration to exploit the best regions.
    # Using the standard deviation of the predicted means as a proxy for global uncertainty.
    mu_std = np.std(mu)
    
    # Scale beta between a minimum and maximum value based on mu_std.
    # If mu_std is large, beta is larger. If mu_std is small, beta is smaller.
    # We map mu_std to a range [1.0, 3.0] roughly.
    # A simple linear scaling or a tanh-like scaling can be used.
    # Let's use a simple scaling: beta = 1.0 + 2.0 * (mu_std / (mu_std + 0.5))
    # This ensures beta is always >= 1.0 and approaches 3.0 as mu_std increases.
    # Alternatively, a fixed beta around 2.0-2.5 often works well, but adaptive might help.
    # Let's try beta = 1.5 + 1.5 * (mu_std / (mu_std + 0.2))
    
    beta = 1.5 + 1.5 * (mu_std / (mu_std + 0.2))
    
    scores = -mu + beta * sigma
    
    # Ensure finiteness
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

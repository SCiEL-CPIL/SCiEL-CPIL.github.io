import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    The score is calculated as kappa * sigma - mu.
    Higher scores correspond to lower predicted means and/or higher uncertainty.
    
    Parameters:
    mu (np.ndarray): Predicted mean values, shape (2000,).
    sigma (np.ndarray): Predicted standard deviations, shape (2000,).
    f_best (float): The best (lowest) function value observed so far.
    
    Returns:
    np.ndarray: Acquisition scores, shape (2000,). Higher values are preferred.
    """
    # Exploration coefficient. 
    # A high value encourages exploration of regions with high uncertainty,
    # which is critical for escaping local minima in the multimodal Hartmann 6D 
    # landscape within a tight 30-evaluation budget.
    kappa = 3.0
    
    # LCB for minimization is mu - kappa * sigma.
    # We want to maximize the acquisition score, so we return the negative:
    # score = -(mu - kappa * sigma) = kappa * sigma - mu
    scores = kappa * sigma - mu
    
    # Ensure the output is finite as required by the task contract.
    # While GP outputs are typically finite, this is a safety check.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

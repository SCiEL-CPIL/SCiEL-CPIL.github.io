import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) 
    with an exploration bonus, normalized by evaluation cost.
    
    Parameters:
    mu (np.array): Mean of the GP predictions, shape (N,)
    sigma (np.array): Standard deviation of the GP predictions, shape (N,)
    f_best (float): The best function value observed so far
    cost (np.array): Evaluation cost for each candidate, shape (N,)
    
    Returns:
    scores (np.array): Acquisition scores, shape (N,)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    if len(mu) != len(sigma) or len(mu) != len(cost):
        return np.zeros(len(mu))
    
    # Small epsilon to avoid division by zero
    eps = 1e-9
    
    # Compute z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # If sigma is very small, z becomes large. 
    # We clip z to avoid numerical issues in norm.cdf/pdf if necessary, 
    # but scipy handles large values well.
    z = (f_best - mu) / (sigma + eps)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Exploration bonus: 
    # We add a term proportional to sigma to encourage exploration of uncertain regions.
    # This helps escape local minima in multimodal functions like Shekel 4d.
    # The weight lambda controls the trade-off.
    # Based on inspiration, a mix is better. Let's use a moderate exploration weight.
    lambda_explore = 0.5
    
    # Combined utility
    utility = ei + lambda_explore * sigma
    
    # Cost-awareness: Divide by cost
    # Cost is in [1, 2]. Lower cost is better.
    # We ensure cost is positive
    cost_safe = np.maximum(cost, eps)
    
    scores = utility / cost_safe
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

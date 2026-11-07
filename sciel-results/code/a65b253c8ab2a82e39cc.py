import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-12)
    
    # Calculate Expected Improvement (EI)
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Using standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    EI = delta * cdf + sigma_safe * pdf
    
    # Handle numerical issues in EI
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    # Cost-aware: use a power-law penalty for cost
    # This encourages exploration of cheaper regions while still
    # considering the expected improvement
    # Using a moderate exponent to balance exploration and exploitation
    cost_penalty = np.power(cost_safe, 0.5)
    scores = EI / cost_penalty
    
    # Also consider a UCB-like component to encourage exploration
    # when uncertainty is high
    beta = 0.5
    ucb_component = beta * sigma_safe
    
    # Combine EI and UCB components
    scores = 0.7 * scores + 0.3 * (ucb_component / cost_penalty)
    
    # Handle any remaining NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

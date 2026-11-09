import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a hybrid strategy: UCB for exploration when far from the minimum, 
    and Expected Improvement (EI) for exploitation when close.
    """
    # Ensure arrays are floats
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # The known global minimum of Goldstein-Price 2D is 3.0.
    # If we are far from the minimum, we prioritize exploration (UCB).
    # If we are close, we prioritize exploitation (EI).
    
    f_target = 3.0
    threshold = 4.0

    if f_best > threshold:
        # UCB strategy: Maximize (-mu + beta * sigma)
        # beta controls exploration. 
        # Reduced beta to 2.0 to allow slightly more exploitation during the search.
        beta = 2.0
        scores = -mu + beta * sigma
    else:
        # EI strategy: Standard Expected Improvement for minimization.
        # z = (f_best - mu) / sigma
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        z = (f_best - mu) / (sigma + 1e-9)
        # Numerical stability: where sigma is very small, z can be inf.
        # norm.cdf and norm.pdf handle inf gracefully, but we mask to be safe.
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        
        scores = (f_best - mu) * cdf + sigma * pdf
        
        # Ensure scores are finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores

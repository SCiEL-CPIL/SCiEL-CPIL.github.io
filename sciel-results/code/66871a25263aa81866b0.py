import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a hybrid acquisition combining Expected Improvement with an
    uncertainty bonus (UCB-like) and a lower confidence bound term.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    N = len(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -20, 20)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # UCB term: encourage exploration of high-uncertainty regions
    # beta controls the trade-off between exploitation and exploration
    # For hartmann 6d, we want moderate exploration
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Lower Confidence Bound: favor points with low predicted values
    lcb = mu - 0.3 * sigma_safe
    
    # Combine terms with weights
    # Normalize each component to have similar scale
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min if ei_range > 0 else 1.0
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    ucb_range = ucb_max - ucb_min if ucb_range > 0 else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    lcb_max = np.max(lcb)
    lcb_min = np.min(lcb)
    lcb_range = lcb_max - lcb_min if lcb_range > 0 else 1.0
    lcb_norm = (lcb - lcb_min) / lcb_range
    
    # Weighted combination: emphasize EI, add exploration via UCB, and exploitation via LCB
    # Higher score = better candidate to evaluate next
    scores = 0.6 * ei_norm + 0.25 * ucb_norm + 0.15 * lcb_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores

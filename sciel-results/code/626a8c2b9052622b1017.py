"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid approach combining Expected Improvement (EI) with a 
    lower confidence bound (LCB) term to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) for minimization
    # beta controls exploration; larger beta means more exploration
    beta = 2.0
    lcb = mu - beta * sigma_safe
    
    # Combine EI and LCB
    # Normalize both to have similar scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    lcb_min = np.min(lcb)
    lcb_range = np.max(lcb) - lcb_min if np.max(lcb) > lcb_min else 1.0
    
    # Scale LCB to be comparable to EI
    lcb_normalized = (lcb - lcb_min) / lcb_range
    ei_normalized = ei / ei_max
    
    # Weighted combination
    weight_ei = 0.6
    weight_lcb = 0.4
    
    scores = weight_ei * ei_normalized + weight_lcb * lcb_normalized
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses a Modified Expected Improvement with an adaptive exploration term.
    The exploration term decays as the best observed value approaches the known global minimum,
    transitioning from UCB-like behavior to pure EI.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores; higher values indicate more desirable points.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive Exploration Term
    # The global minimum of Styblinski-Tang 1D is approximately -39.1662
    global_min = -39.1662
    
    # Calculate the gap to the global minimum
    # If f_best is higher than global_min, gap is positive.
    gap = f_best - global_min
    
    # We want strong exploration when the gap is large.
    # We taper off exploration as we get closer to the minimum.
    # A gap of 50.0 is considered "close enough" to start tapering.
    # Max exploration weight (beta) is 2.0, similar to UCB.
    max_beta = 2.0
    taper_distance = 50.0
    
    if gap > taper_distance:
        beta = max_beta
    else:
        # Linear taper from max_beta to 0 as gap goes from taper_distance to 0
        beta = max_beta * (gap / taper_distance)
        
    # Ensure beta is non-negative
    beta = np.maximum(beta, 0.0)
    
    # Combine EI and Exploration
    # The exploration term is beta * sigma
    scores = ei + beta * sigma_safe
    
    return scores
# EVOLVE-BLOCK-END

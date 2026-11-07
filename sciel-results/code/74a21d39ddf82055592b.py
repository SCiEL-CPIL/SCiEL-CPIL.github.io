"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for minimization using Expected Improvement (EI)
    with a small exploration bonus to handle multimodal landscape.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    # For minimization, improvement is f_best - mu (positive if mu < f_best)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Expected Improvement for minimization:
    # EI = sigma * (z * phi(z) + Phi(z)) where phi is PDF and Phi is CDF of standard normal
    # Using scipy.special for accuracy
    from scipy.special import ndtr, ndtri
    from scipy.stats import norm
    
    # Compute EI
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = sigma * (z * phi_z + Phi_z)
    
    # For points where mu >= f_best, EI is 0 or negative; clip to 0
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus to help escape local minima
    # This is a soft UCB-like term that encourages exploration in uncertain regions
    # but is much smaller than the exploitation term
    exploration_bonus = 0.3 * sigma
    
    # Combine: primary signal is EI, with a small exploration boost
    scores = ei + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

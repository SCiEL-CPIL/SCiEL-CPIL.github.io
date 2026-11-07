"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a
    Lower Confidence Bound (LCB) exploration bonus.
    
    Parameters
    ----------
    mu : np.ndarray
        GP mean predictions at candidate points, shape (N,)
    sigma : np.ndarray
        GP standard deviation predictions at candidate points, shape (N,)
    f_best : float
        Best observed value so far (minimum value for minimization)
    
    Returns
    -------
    np.ndarray
        Acquisition scores for each candidate, shape (N,).
        Higher scores indicate more promising candidates.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) term: favors points with high uncertainty
    # LCB = -mu - beta * sigma  (we want to maximize this, so we add it)
    # Using a moderate exploration constant
    beta = 2.0
    lcb_bonus = beta * sigma_safe
    
    # Combine EI and LCB exploration
    # We scale EI to be comparable to the LCB term
    # EI is non-negative, LCB bonus is non-negative
    # We use a weighted combination
    acquisition_scores = ei + lcb_bonus
    
    # Ensure all scores are finite
    acquisition_scores = np.where(np.isfinite(acquisition_scores), acquisition_scores, 0.0)
    
    return acquisition_scores
# EVOLVE-BLOCK-END

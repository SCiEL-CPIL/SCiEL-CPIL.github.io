"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and a small exploration
    term to balance exploitation and exploration. The exploration term is
    based on the standard deviation to encourage visiting uncertain regions.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores for each candidate point.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_clipped = np.clip(sigma, 1e-9, None)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_clipped
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima and encourages exploration
    exploration_bonus = 0.1 * sigma_clipped
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

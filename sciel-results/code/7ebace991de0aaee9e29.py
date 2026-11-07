"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    
    Uses a combination of Expected Improvement (EI) and a diversity/exploration
    term based on uncertainty to balance exploitation and exploration.
    The diversity term helps escape local optima by encouraging exploration
    in regions of high uncertainty, while EI drives the search toward promising
    regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (n_candidates,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores for each candidate. Higher scores indicate more
        promising candidates to evaluate next.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute a diversity/exploration term
    # This encourages exploration in regions of high uncertainty
    # We use a scaled version of sigma to add exploration
    # The weight is chosen to balance exploitation and exploration
    diversity_weight = 0.1
    diversity_term = diversity_weight * sigma
    
    # Combine EI and diversity term
    scores = ei + diversity_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

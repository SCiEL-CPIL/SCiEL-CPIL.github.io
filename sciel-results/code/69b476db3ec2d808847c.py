"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    EI is a deterministic acquisition that balances exploration and exploitation
    by computing the expected improvement over the current best value f_best.
    It has a well-defined theoretical basis and is known to perform well for
    multi-modal functions like the Michalewicz function.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means evaluate next.
    """
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Standardized improvement: (f_best - mu) / sigma
    # Positive when mu < f_best (potential improvement)
    z = (f_best - mu) / sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    from scipy.stats import norm
    
    # Compute the standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure non-negative EI (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

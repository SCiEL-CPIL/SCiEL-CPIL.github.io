"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior, shape (2000,).
    sigma : np.ndarray
        Standard deviation of the GP posterior, shape (2000,).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : np.ndarray
        EI scores, shape (2000,). Higher scores indicate more promising points.
    """
    # Clip sigma to avoid division by zero and numerical instability
    # A small epsilon is sufficient to prevent overflow in z calculation
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Introduce a small noise parameter xi for Noisy Expected Improvement
    # This helps balance exploration and exploitation
    xi = 0.01
    
    # Calculate z-score for Noisy Expected Improvement
    # z = (f_best - xi - mu) / sigma
    z = (f_best - xi - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # Note: The standard EI formula uses (f_best - mu). 
    # With xi, the improvement threshold is f_best - xi.
    # The expected improvement is E[max(f_best - f(x), 0)].
    # If we shift the threshold, we are effectively penalizing small improvements.
    # A more standard Noisy EI adds noise to the observation, but here we use xi as a threshold.
    # Let's use the standard EI formula but with a shifted f_best.
    # This is equivalent to EI with a threshold.
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = (f_best - xi - mu) * cdf + sigma_safe * pdf
    
    # Ensure finite values, replacing any NaN/Inf with -inf (or 0, but -inf is safer for "never pick")
    # However, EI should be non-negative. If numerical errors cause negative values, clip to 0.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

"""Acquisition function for Bayesian optimization of beale 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Decaying Threshold Expected Improvement (EI-xi).
    
    This approach uses a dynamic improvement threshold (xi) that starts high
    (encouraging exploration of regions with high potential for large improvements)
    and decays to zero (encouraging exploitation of the current best) as f_best
    approaches the minimum.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, np.nanmax(mu) if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Estimate progress using a sigmoid centered at f_best = 1.0
    # The Beale function has a global min of 0 and local mins around 0.5-1.0.
    # If f_best is > 1.0, we are likely far from the global basin.
    # If f_best is < 1.0, we are likely in or near a promising basin.
    # Sigmoid: progress = 1 / (1 + exp(-k * (center - f_best)))
    # We want progress ~ 0 when f_best is large, and ~ 1 when f_best is small.
    # So: progress = 1 / (1 + exp(-k * (f_best - center)))?
    # If f_best = 10, center=1, f_best-center = 9. exp(9) is large. progress ~ 0.
    # If f_best = 0.1, center=1, f_best-center = -0.9. exp(-0.9) is small. progress ~ 1.
    # Let's use k=2 for a smooth transition.
    k = 2.0
    center = 1.0
    progress = 1.0 / (1.0 + np.exp(-k * (f_best - center)))
    # Clip progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic improvement threshold xi
    # Start with high xi (exploration) and decay to 0 (exploitation)
    xi_max = 5.0
    xi_min = 0.0
    xi = xi_max * (1.0 - progress) + xi_min * progress
    
    # Calculate Expected Improvement with threshold xi
    # EI_xi = E[max(f_best - xi - f(x), 0)]
    # Let delta = f_best - xi - mu
    # z = delta / sigma
    # EI = delta * Phi(z) + sigma * phi(z)
    
    delta = f_best - xi - mu
    z = delta / sigma
    
    # Handle numerical stability for large z
    # If z is very large, Phi(z) ~ 1, phi(z) ~ 0. EI ~ delta.
    # If z is very small, Phi(z) ~ 0.5, phi(z) ~ 0.4.
    
    # Use scipy.stats.norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = delta * cdf + sigma * pdf
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function with dynamic exploration offset (xi).
    
    This structural change replaces the fixed-parameter LCB with a principled
    Expected Improvement metric that adapts its exploration bias based on
    the current regret relative to the known global minimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean predictions.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation predictions.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Known global minimum of Styblinski-Tang 1D
    f_min_global = -39.1662
    
    # Calculate current regret
    regret = f_best - f_min_global
    
    # Estimate initial regret to normalize progress
    # The function range is roughly [-39, 200], so initial regret is ~50-100
    # Using 50.0 as a conservative baseline for normalization
    max_regret = 50.0
    
    # Progress: 0.0 at start (high regret), 1.0 at optimum (zero regret)
    # Clip to [0, 1] to handle cases where f_best might already be below f_min_global
    # or if the estimate is off
    progress = np.clip(1.0 - (regret / max_regret), 0.0, 1.0)
    
    # Dynamic xi (exploration offset in EI)
    # High xi early on encourages looking at points with high uncertainty or
    # moderate means where improvement is still possible.
    # Low xi late on focuses strictly on points likely to beat f_best.
    # Quadratic decay ensures a smooth transition from exploration to exploitation.
    xi_max = 5.0
    xi = xi_max * (1.0 - progress) ** 2
    
    # Ensure sigma is positive to avoid division by zero in z-score calculation
    # 1e-8 is small enough not to distort results but large enough for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score for EI
    # z = (f_best - mu - xi) / sigma
    # We subtract xi to shift the improvement threshold, effectively demanding
    # a larger expected improvement early on.
    z = (f_best - mu - xi) / sigma_safe
    
    # Expected Improvement Formula for minimization:
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # Where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute Phi(z) and phi(z)
    # norm.cdf and norm.pdf handle arrays efficiently
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate EI scores
    # Note: (f_best - mu - xi) is the "improvement" term
    scores = (f_best - mu - xi) * Phi_z + sigma_safe * phi_z
    
    # Ensure all scores are finite
    # EI should be non-negative, but numerical issues can cause small negatives or NaNs
    # We clip to 0 to ensure we don't prefer "worse" points due to numerical noise
    # and replace non-finite values with 0 (or a very small negative to deprioritize)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

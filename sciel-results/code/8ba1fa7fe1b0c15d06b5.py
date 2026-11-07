"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement (EI) acquisition function.
    
    Uses a dynamic xi parameter that decreases as optimization progresses,
    shifting from exploration to exploitation. Also handles the case where
    sigma is very small (near convergence) gracefully.
    """
    n_candidates = len(mu)
    
    # Estimate progress: we don't know exact eval count, but we can infer
    # from the spread of mu and sigma. Early on, sigma is large relative to range.
    # We'll use a simple heuristic: if sigma has low variance, we're likely late-stage.
    
    # Compute a rough measure of exploration needed
    # If f_best is close to the known minimum (0 for Beale), we should exploit more
    # The Beale function minimum is 0 at (3, 0.5)
    
    # Adaptive xi: start high for exploration, decrease as we get closer to minimum
    # Use the ratio of current best to a reference scale
    # Since we don't know the eval count, use the spread of predictions
    
    # A robust approach: use the median sigma as a proxy for uncertainty level
    sigma_med = np.median(sigma)
    mu_range = np.max(mu) - np.min(mu)
    
    # If uncertainty is high relative to function range, explore more
    # Normalize sigma to get a relative uncertainty measure
    if mu_range > 1e-10:
        rel_uncertainty = sigma_med / (mu_range + 1e-10)
    else:
        rel_uncertainty = 0.5
    
    # xi controls the "optimism" of EI
    # High xi -> more exploration (willing to accept points that might not beat f_best)
    # Low xi -> more exploitation (only accept points likely to beat f_best)
    # Map rel_uncertainty to xi in [0.01, 0.5]
    xi = 0.01 + 0.49 * min(rel_uncertainty, 1.0)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # For a GP: EI(x) = (f_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x) - xi) / sigma(x)
    
    # Handle sigma = 0 cases
    sigma_safe = np.maximum(sigma, 1e-10)
    
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Where sigma was essentially 0, EI should be 0 if mu >= f_best, else small positive
    mask_low_sigma = sigma < 1e-10
    if np.any(mask_low_sigma):
        ei[mask_low_sigma] = np.maximum(f_best - mu[mask_low_sigma], 0) * 0.1
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

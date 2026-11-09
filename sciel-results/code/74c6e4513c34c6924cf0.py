"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with dynamic exploration constant.
    Uses a proxy for the optimization stage based on the distribution of posterior
    standard deviations to adjust the exploration-exploitation trade-off.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far (lowest since we minimize).
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        UCB scores; higher values indicate more promising candidates.
    """
    # Sanitize inputs to ensure finite values
    mu = np.where(np.isfinite(mu), mu, -np.inf)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Calculate uncertainty metrics to estimate the stage of optimization
    sigma_max = np.max(sigma)
    sigma_mean = np.mean(sigma)
    
    if sigma_max < 1e-8:
        # If uncertainty is negligible, exploit the mean
        scores = mu
    else:
        # Ratio of mean to max sigma:
        # Early stage: sigma is relatively uniform and high -> ratio close to 1
        # Late stage: sigma has localized peaks and low valleys -> ratio decreases
        sigma_ratio = sigma_mean / sigma_max
        
        # Dynamic kappa:
        # We map the ratio [0, 1] to a kappa range [1.0, 5.0].
        # Higher kappa (more exploration) when ratio is high (early).
        # Lower kappa (more exploitation) when ratio is low (late).
        # This range is more aggressive than typical defaults to ensure
        # coverage of the multimodal Hartmann 3D landscape.
        kappa = 1.0 + 4.0 * sigma_ratio
        
        # Compute UCB scores for minimization: mu - kappa * sigma
        scores = mu - kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

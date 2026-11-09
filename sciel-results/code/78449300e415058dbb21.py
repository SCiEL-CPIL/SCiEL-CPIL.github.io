"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Samples from the posterior distribution at each candidate point and
    selects the point with the lowest sampled value (for minimization).
    Since the evaluator selects the candidate with the highest score,
    we return the negative of the sampled values.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far. (Not used directly in TS, but kept for signature)
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is non-negative and avoid division by zero if we were doing z-scores,
    # but here we just multiply. Still, clip to avoid numerical issues if sigma is negative due to error.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw random standard normal variables for each candidate
    z = np.random.randn(len(mu))
    
    # Sample from the posterior: x ~ N(mu, sigma^2)
    # x = mu + z * sigma
    samples = mu + z * sigma_safe
    
    # We want to minimize the function, so we prefer lower samples.
    # The evaluator picks the max score.
    # So we return -samples.
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

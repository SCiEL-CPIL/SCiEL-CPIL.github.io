"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Minimum variance acquisition function.
    
    This strategy prioritizes exploration by selecting the point with the highest
    posterior standard deviation. This forces the Gaussian Process to resolve
    uncertainty in unexplored regions, which can prevent premature convergence
    to local minima or noise, especially in the early stages of optimization.
    
    Args:
        mu: Array of shape (2000,) with posterior means.
        sigma: Array of shape (2000,) with posterior standard deviations.
        f_best: The lowest observed value so far.
        
    Returns:
        Array of shape (2000,) with acquisition scores. Higher is better.
    """
    # The directive specifies scoring solely by posterior standard deviation.
    # We return sigma directly, as higher sigma indicates higher uncertainty.
    
    # Ensure the output is finite and valid.
    # sigma should be non-negative, but we sanitize to be safe.
    scores = np.asarray(sigma)
    
    # Replace any NaN or Inf values with 0.0 to ensure finite scores.
    # This handles potential numerical issues in the GP computation.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

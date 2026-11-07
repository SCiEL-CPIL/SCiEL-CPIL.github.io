"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB) for minimization.
    
    This function uses a pure UCB strategy with a high exploration parameter
    to navigate the multimodal landscape of the Michalewicz function.
    
    Args:
        mu: (N,) array of posterior means.
        sigma: (N,) array of posterior standard deviations.
        f_best: float, the best (lowest) observed value so far.
        
    Returns:
        (N,) array of acquisition scores. Higher is better.
    """
    # Ensure sigma is non-negative and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB for minimization:
    # We want to maximize the score to select the next point.
    # A point is good if its mean is low (exploit) or its uncertainty is high (explore).
    # Score = -mu + beta * sigma
    
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta favors exploration, which is crucial for multimodal functions
    # to avoid getting stuck in local minima.
    # Based on findings, high exploration helps in small budget regimes.
    beta = 2.0
    
    score = -mu + beta * sigma_safe
    
    # Ensure the output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

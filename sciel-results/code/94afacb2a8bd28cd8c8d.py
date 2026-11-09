"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean of the Gaussian Process at candidate points.
    sigma (np.ndarray): Posterior standard deviation of the Gaussian Process at candidate points.
    f_best (float): The best observed function value so far.
    
    Returns:
    np.ndarray: Scores for the candidate points. Higher scores indicate more promising points.
    """
    # UCB for minimization: We want to minimize f(x).
    # We select points that have low mean (good) and high uncertainty (exploration).
    # The objective to minimize is mu(x) - beta * sigma(x).
    # Since we need to return scores to be MAXIMIZED, we negate the objective:
    # score = -(mu(x) - beta * sigma(x)) = beta * sigma(x) - mu(x)
    
    beta = 1.5
    
    # Ensure sigma is non-negative (it should be, but just in case of numerical noise)
    sigma = np.maximum(sigma, 0.0)
    
    scores = beta * sigma - mu
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

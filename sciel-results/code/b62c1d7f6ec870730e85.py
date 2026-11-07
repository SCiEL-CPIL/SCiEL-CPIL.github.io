"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function.
    
    This strategy directly maximizes the lower bound of the posterior,
    balancing exploitation (low mu) and exploration (high sigma).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    sigma = np.maximum(sigma, 1e-12)
    
    # LCB for minimization: score = mu - beta * sigma
    # Higher score means lower predicted value with penalty for uncertainty.
    # Beta controls the exploration-exploitation trade-off.
    # A value around 1.0-2.0 is typical. 
    # Since sigma is in objective units and mu is in objective units, 
    # beta is a dimensionless multiple of standard deviations.
    # Branin has multiple minima. Early on, we want to explore.
    # As f_best drops, we might want to exploit more, but LCB inherently
    # does this because mu drops in promising areas.
    # Let's use a fixed beta of 1.5, which is a standard choice.
    beta = 1.5
    
    scores = mu - beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

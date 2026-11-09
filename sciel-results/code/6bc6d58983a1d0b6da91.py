"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Upper Confidence Bound (UCB) for minimization.
    
    UCB balances exploration and exploitation by adding an exploration bonus
    proportional to the standard deviation. For minimization, we want to find
    points where the predicted mean is low AND the uncertainty is high.
    
    Formula:
    UCB = mu - kappa * sigma
    
    Where kappa controls the exploration-exploitation tradeoff. A larger kappa
    encourages more exploration of uncertain regions.
    
    We use a moderately aggressive kappa to balance finding new minima with
    exploiting known good regions.
    """
    # Exploration-exploitation tradeoff parameter
    kappa = 2.0
    
    # Calculate UCB score (higher is better for selection)
    # For minimization, lower mu is better, so we use -mu + kappa * sigma
    # This means we prefer low predicted values and high uncertainty
    ucb = -mu + kappa * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(ucb), ucb, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

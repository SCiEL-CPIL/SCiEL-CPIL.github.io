# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a cost-adjusted LCB formulation.
    
    The exploration term is scaled by 1/cost so that high-cost regions are
    penalized more heavily in the uncertainty bonus, while the mean term
    remains invariant. This creates a non-uniform exploration landscape that
    prioritizes high-uncertainty, low-cost areas early in the budget.
    """
    # Clip sigma to avoid division by zero and ensure numerical stability
    sigma = np.maximum(sigma, 1e-5)
    
    # Cost is 1 + mean(x scaled to [0,1]), so cost is in [1, 2]
    # Ensure cost is at least 1e-9 to avoid division issues
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive beta:
    # When f_best is large (poor initial estimates), we want high beta (exploration).
    # When f_best is small (close to optimum), we want lower beta (exploitation).
    # beta = base + range * (f_best / (f_best + scale))
    # base = 2.0, range = 1.0, scale = 1.0
    # If f_best = 0, beta = 2.0
    # If f_best = 1, beta = 2.5
    # If f_best = 10, beta = ~2.9
    # If f_best = 100, beta = ~3.0
    
    beta = 2.0 + 1.0 * (f_best / (f_best + 1.0))
    
    # Cost-adjusted LCB acquisition
    # Minimize: mu - beta * sigma / cost
    # Maximize: -mu + beta * sigma / cost
    acquisition_scores = -mu + beta * sigma / cost
    
    # Ensure all scores are finite
    acquisition_scores = np.where(np.isfinite(acquisition_scores), acquisition_scores, -np.inf)
    
    return acquisition_scores
# EVOLVE-BLOCK-END

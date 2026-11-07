"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization minimization.
    
    Combines Expected Improvement (EI) with a UCB exploration term,
    all normalized by cost^gamma to strongly favor low-cost evaluations.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Parameters
    gamma = 3.0
    kappa = 2.0
    
    # Cost-normalized UCB strategy for minimization
    # Maximizes (-mu + kappa * sigma) to favor low predicted values and high uncertainty
    # Divided by cost^gamma to drastically favor low-cost evaluations
    ucb_score = (-mu + kappa * sigma_safe)
    
    # Normalize by cost^gamma
    final_score = ucb_score / (cost_safe ** gamma)
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END

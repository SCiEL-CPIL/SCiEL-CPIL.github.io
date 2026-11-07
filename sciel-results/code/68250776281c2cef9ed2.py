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
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB score for minimization: (-mu + kappa * sigma) gives higher scores
    # to points with low predicted value and high uncertainty
    ucb_score = (-mu + kappa * sigma_safe)
    
    # Combine EI and UCB with weights
    # EI is in objective units, UCB is also in objective units
    # Use EI as primary exploitation signal, UCB for exploration
    combined = 0.5 * ei + 0.5 * ucb_score
    
    # Normalize by cost^gamma to strongly favor low-cost evaluations
    final_score = combined / (cost_safe ** gamma)
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    UCB balances exploitation (low mu) with exploration (high sigma).
    """
    # Handle potential numerical issues with sigma
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # UCB for minimization: minimize mu - kappa * sigma
    # Higher score means more promising (lower predicted value with uncertainty bonus)
    # We want to return scores where higher is better for evaluation
    # So we return -mu + kappa * sigma (negating the objective)
    
    # Adaptive kappa based on number of evaluations
    # Since we don't have access to n directly, use a fixed reasonable value
    # For 30 total evaluations, kappa around 1.0-2.0 works well
    kappa = 1.5
    
    # UCB score: -mu + kappa * sigma
    # Higher values indicate points that are either promising (low mu) or uncertain (high sigma)
    ucb_score = -mu + kappa * safe_sigma
    
    # Ensure all scores are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    return ucb_score
# EVOLVE-BLOCK-END

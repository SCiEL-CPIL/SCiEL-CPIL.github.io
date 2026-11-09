"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition function for minimization.
    
    Maximizes the predicted 5th percentile of the posterior:
    score = -(mu - 1.645 * sigma) = -mu + 1.645 * sigma
    
    This directly targets the lower tail of the predictive distribution,
    providing an aggressive exploitation signal well-suited for the
    unimodal 1D sphere function.
    """
    # Ensure sigma is non-negative and handle near-zero values
    sigma = np.maximum(sigma, 1e-10)
    
    # Fixed quantile coefficient: z_{0.05} = -1.645
    # For minimization, we maximize the 5th percentile = mu + 1.645 * sigma
    # But since we want lower values, we use -mu + 1.645 * sigma as the score
    # (higher score means more promising)
    
    score = -mu + 1.645 * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

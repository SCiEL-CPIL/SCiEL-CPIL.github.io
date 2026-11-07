"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a small
    exploration bonus proportional to the standard deviation (sigma).
    
    This hybrid approach balances exploitation (finding points likely to
    improve the current best) with exploration (visiting uncertain regions),
    which is crucial for the limited 30-evaluation budget on Hartmann 6D.
    """
    # Ensure sigma is strictly positive to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for the normal distribution
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term to encourage visiting high-uncertainty areas
    # The weight 0.1 is a heuristic chosen to provide a modest boost to exploration
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    # Total score: higher is better
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

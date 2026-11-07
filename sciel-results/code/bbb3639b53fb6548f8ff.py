"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) for exploitation
    2. Knowledge Gradient-inspired term for exploration
    3. Adaptive beta based on uncertainty and progress
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher means evaluate next
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle edge cases where EI might be NaN or inf
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Knowledge Gradient term: prefer points with high uncertainty
    # but penalize points that are already well-explored
    kg_term = sigma * np.exp(-sigma / (np.std(sigma) + 1e-8))
    
    # Adaptive exploration-exploitation balance
    # Use the ratio of best improvement potential to current best
    improvement_potential = f_best - np.min(mu)
    if improvement_potential > 1e-8:
        # More exploration when there's still significant improvement potential
        beta = 0.5 + 0.5 * (improvement_potential / (abs(f_best) + 1e-8))
        beta = np.clip(beta, 0.3, 1.0)
    else:
        beta = 0.3
    
    # Combine EI and KG terms
    # Normalize EI to [0, 1] range for better combination
    ei_max = np.max(ei)
    if ei_max > 1e-8:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    # Normalize KG term
    kg_max = np.max(kg_term)
    if kg_max > 1e-8:
        kg_normalized = kg_term / kg_max
    else:
        kg_normalized = kg_term
    
    # Final score: weighted combination
    scores = (1.0 - beta) * ei_normalized + beta * kg_normalized
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END

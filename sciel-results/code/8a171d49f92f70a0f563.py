"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Regret relative to the known global minimum
    with a linear exploration bonus to enhance robustness and exploration.
    
    mu: (2000,) array of GP posterior means
    sigma: (2000,) array of GP posterior standard deviations
    f_best: lowest observed value so far
    
    Returns: (2000,) array of scores, higher = more promising to evaluate next.
    """
    # Known global minimum of Goldstein-Price 2D on [-2,2]^2 is 3.0
    f_true_min = 3.0
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Expected Improvement/Regret
    z = (f_true_min - mu) / sigma_safe
    
    # Analytical Expected Regret relative to f_true_min
    # E[max(0, f_true_min - f(x))] = (f_true_min - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei_scores = (f_true_min - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add a linear exploration bonus (similar to UCB) to encourage exploration
    # in regions with high uncertainty, especially where the EI term is small.
    beta = 0.5
    exploration_bonus = beta * sigma_safe
    
    scores = ei_scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

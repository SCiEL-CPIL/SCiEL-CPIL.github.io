"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI)
    and a UCB-like exploration score.
    
    - EI favors points with high expected improvement over f_best.
    - UCB-like score favors points with low predicted mean (mu) and high uncertainty (sigma).
    
    The scores are normalized to comparable ranges before combination.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute UCB-like Exploration Score
    # We want to reward low mu (exploitation potential) and high sigma (exploration potential).
    # Normalize mu and sigma to [0, 1] to ensure they are on a similar scale.
    
    # Normalize mu: 1.0 for minimum mu, 0.0 for maximum mu
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-12:
        mu_norm = np.ones_like(mu)
    else:
        mu_norm = (mu_max - mu) / mu_range
        
    # Normalize sigma: 1.0 for maximum sigma, 0.0 for minimum sigma
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-12:
        sigma_norm = np.ones_like(sigma)
    else:
        sigma_norm = (sigma - sigma_min) / sigma_range
        
    # Combine exploration components: weight low mu higher than high sigma
    # This creates a score in [0, 1.5]
    ucb_score = mu_norm + 0.5 * sigma_norm
    
    # 3. Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
        
    # 4. Combine EI and UCB scores
    # Weights chosen to balance exploitation (EI) and exploration (UCB)
    # Based on Inspiration 1 which showed improvement over pure EI
    w_ei = 0.8
    w_ucb = 0.2
    
    score = w_ei * ei_norm + w_ucb * ucb_score
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

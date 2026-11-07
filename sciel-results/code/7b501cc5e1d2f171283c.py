# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses an adaptive Upper Confidence Bound (UCB) with a small EI component
    to balance exploration and exploitation.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive UCB parameter
    # Start with high exploration, reduce as we find better values
    # Use the mean uncertainty to scale the exploration
    mean_sigma = np.mean(sigma_safe)
    max_sigma = np.max(sigma_safe)
    
    # Adaptive kappa: higher when uncertainty is high, lower when we're close to optimum
    # The term (f_best - np.min(mu)) / (max_sigma + 1e-9) measures how far we are from the best possible
    # We use a sigmoid-like function to smoothly transition
    improvement_potential = (f_best - np.min(mu)) / (max_sigma + 1e-9)
    base_kappa = 1.0 + 2.0 * np.exp(-improvement_potential)
    
    # Scale kappa by the relative uncertainty
    relative_uncertainty = mean_sigma / (max_sigma + 1e-9)
    kappa = base_kappa * (0.5 + 0.5 * relative_uncertainty)
    
    # UCB for minimization
    ucb = mu - kappa * sigma_safe
    
    # Combine UCB with a small EI component
    # EI is already in the right direction (higher is better for improvement)
    # Scale EI to be comparable to UCB
    ei_scale = np.max(np.abs(ucb)) + 1e-9
    scaled_ei = ei / ei_scale
    
    # Final score: primarily UCB, with small EI contribution
    score = ucb + 0.1 * scaled_ei * (np.max(np.abs(ucb)) + 1e-9)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

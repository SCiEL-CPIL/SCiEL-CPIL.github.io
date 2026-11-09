"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4d.
    
    Uses a blend of Expected Improvement (EI) and Probability of Improvement (PoI)
    normalized by evaluation cost. The blend balances exploitation (EI) with
    exploration (PoI) to navigate the multimodal landscape effectively.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI: expected improvement over f_best
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute PoI: probability of improvement
    poi = norm.cdf(z)
    
    # Scale EI to be comparable to PoI (which is in [0,1])
    # Use a characteristic scale based on current uncertainty
    # This helps balance the two terms properly
    scale = np.maximum(np.abs(f_best - mu), sigma_safe)
    ei_scaled = ei / scale
    
    # Blend EI and PoI with adaptive weighting
    # When uncertainty is high (sigma large), favor PoI (exploration)
    # When uncertainty is low (sigma small), favor EI (exploitation)
    # Use a sigmoid-like transition based on relative uncertainty
    rel_uncertainty = sigma_safe / (np.abs(f_best - mu) + sigma_safe + 1e-9)
    
    # Weight: higher when uncertainty is high
    # alpha = 0.5 * (1 + tanh(2 * (rel_uncertainty - 0.5)))
    # This gives alpha in [0,1], ~0 when rel_uncertainty < 0.5, ~1 when > 0.5
    alpha = 0.5 * (1 + np.tanh(4 * (rel_uncertainty - 0.5)))
    
    # Blend: alpha * EI_scaled + (1 - alpha) * PoI
    blended = alpha * ei_scaled + (1 - alpha) * poi
    
    # Normalize by cost (higher cost = lower priority)
    cost_safe = np.maximum(cost, 1e-9)
    score = blended / cost_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

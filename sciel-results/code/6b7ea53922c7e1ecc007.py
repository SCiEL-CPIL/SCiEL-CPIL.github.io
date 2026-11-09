# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid Expected Improvement with
    cost normalization and adaptive exploration.
    
    Key structural change: Instead of log-space multiplicative model, we use
    a direct Expected Improvement (EI) formulation normalized by cost, with
    an adaptive kappa that increases with uncertainty to balance exploration
    and exploitation more effectively.
    
    The EI is computed as:
    E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma, Phi is standard normal CDF, phi is PDF.
    
    We then normalize by cost to get expected improvement per unit cost.
    """
    eps = 1e-12
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score: how many std devs f_best is above mu
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    # Using scipy.special for accurate computation
    try:
        from scipy.special import ndtr, ndtri
        Phi = ndtr(z)  # CDF
        phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)  # PDF
    except ImportError:
        # Fallback: approximate CDF and PDF
        Phi = 0.5 * (1 + np.tanh(z / np.sqrt(2)))  # rough approximation
        phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (it should be by construction, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: expected improvement per unit cost
    # cost is in range [1, 2]
    cost_safe = np.maximum(cost, eps)
    
    # Use a power-based cost penalty to control the strength of cost awareness
    # beta=1.0 means linear normalization (EI / cost)
    # beta=0.5 means sqrt normalization (EI / sqrt(cost)) - less aggressive
    beta = 1.0
    score = ei / (cost_safe ** beta)
    
    # Add a small exploration bonus based on uncertainty
    # This helps when EI is zero (mu >= f_best) but sigma is high
    # The bonus is scaled by sigma to reward truly uncertain regions
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    score = score + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END

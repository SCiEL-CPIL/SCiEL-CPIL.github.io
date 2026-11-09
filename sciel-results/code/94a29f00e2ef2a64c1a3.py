"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a dynamic Pareto-based weighted sum of
    normalized Expected Improvement (EI) and normalized Uncertainty (UCB).
    
    The weight shifts from exploration (UCB) to exploitation (EI) based on
    the current average predictive uncertainty.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Small epsilon for numerical stability
    eps = 1e-9
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Uncertainty term (UCB-like)
    # We use a standard UCB: -mu + beta * sigma
    # Using beta=1.0 provides a balance between low mean and high uncertainty.
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # 3. Normalize EI and UCB to [0, 1] range
    # This ensures the terms are comparable in magnitude
    
    # Normalize EI
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < eps:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    # Normalize UCB
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < eps:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
        
    # 4. Compute Dynamic Weight
    # If average sigma is high, we are likely in the early stages -> prioritize UCB (w small)
    # If average sigma is low, we are likely in the later stages -> prioritize EI (w large)
    avg_sigma = np.mean(sigma)
    
    # Sigmoid-like transition for the weight
    # We want w to be close to 0 when avg_sigma is large, and close to 1 when avg_sigma is small.
    # A simple inverse relationship works well:
    # w = 1 / (1 + avg_sigma)
    # However, the scale of sigma depends on the GP noise and length scales.
    # A more robust approach is to normalize the weight based on the current uncertainty spread.
    
    # Let's use a scaled version. If avg_sigma is 0.1, w = 1/(1+0.1) ~ 0.9 (Exploit)
    # If avg_sigma is 1.0, w = 1/(1+1.0) ~ 0.5 (Balanced)
    # If avg_sigma is 10.0, w = 1/(1+10.0) ~ 0.09 (Explore)
    
    # To make it more adaptive, we can map the average sigma to a weight using a scaling factor.
    # Given the Hartmann 6D problem and standard GP settings, sigma is often O(1) initially.
    # Let's use:
    w = 1.0 / (1.0 + avg_sigma)
    
    # Clip weight to [0.1, 0.9] to prevent complete dominance of one term
    # This ensures we always consider both improvement and uncertainty to some extent
    w = np.clip(w, 0.1, 0.9)
    
    # 5. Combine Scores
    # Higher score is better
    # We want to maximize the weighted sum
    score = w * ei_norm + (1.0 - w) * ucb_norm
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

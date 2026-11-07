import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses a hybrid approach: Variance-Gated Expected Improvement (VEI)
    combined with a small UCB exploration term, with a time guard.
    """
    start_time = time.time()
    deadline = 300.0  # 300s limit, leave margin
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Variance-Gated EI: scale EI by (sigma / sigma_max)^gamma
    # This penalizes exploitation in low-uncertainty regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        sigma_ratio = sigma_safe / sigma_max
    else:
        sigma_ratio = np.ones_like(sigma_safe)
    
    gamma = 1.5
    vei = ei * (sigma_ratio ** gamma)
    
    # Add a small UCB exploration term to differentiate among top candidates
    # UCB for minimization: beta * sigma - mu
    # We use a small beta to avoid over-exploration
    beta = 0.1
    ucb_term = beta * sigma_safe - mu
    
    # Combine: primarily VEI, with a small UCB component
    # Scale UCB term relative to VEI magnitude
    vei_scale = np.max(np.abs(vei)) + eps
    ucb_scaled = ucb_term / vei_scale
    
    scores = vei + 0.1 * ucb_scaled
    
    # Check time
    if time.time() - start_time > deadline - 5.0:
        # Return best result found with safety margin
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

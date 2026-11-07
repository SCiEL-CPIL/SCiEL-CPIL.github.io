import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Base UCB Score (Proven best: kappa=1.0)
    kappa = 1.0
    ucb_score = -mu + kappa * sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Handle division by zero in sigma
    sigma_safe = np.where(sigma < 1e-10, 1.0, sigma)
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    phi_cdf_z = norm.cdf(z)
    
    ei_scores = (f_best - mu) * phi_cdf_z + sigma_safe * phi_z
    
    # For points where sigma was 0 (or very small), correct the score
    # If sigma < 1e-10, the variance is negligible.
    # The improvement is simply max(0, f_best - mu)
    mask_low_sigma = sigma < 1e-10
    ei_scores[mask_low_sigma] = np.maximum(0, f_best - mu[mask_low_sigma])
    
    # Ensure finite values
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # Hybrid Score: Primarily UCB (which scored 1.54) with a small EI boost (1.0)
    # The EI term helps in finely exploiting regions where UCB might be flat or misleading
    # due to high uncertainty far from the optimum.
    # We weight UCB heavily to preserve the exploration advantage.
    alpha = 0.9
    beta = 0.1
    
    score = alpha * ucb_score + beta * ei_scores
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return score
        
    return score

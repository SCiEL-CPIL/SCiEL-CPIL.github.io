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
    
    # Adaptive kappa: 
    # We want MORE exploration (higher kappa) when f_best is LARGE (early, far from optimum).
    # We want LESS exploration (lower kappa) when f_best is SMALL (late, close to optimum).
    # Current code had this inverted.
    # Using tanh for a smooth, bounded increase.
    # f_best starts ~5-10, goes to 0.
    # tanh(0) = 0, tanh(5) ~ 1.
    kappa = 1.0 + 1.5 * np.tanh(f_best)
    
    ucb_score = -mu + kappa * sigma
    
    # Calculate Expected Improvement (EI) for minimization
    sigma_safe = np.where(sigma < 1e-10, 1.0, sigma)
    z = (f_best - mu) / sigma_safe
    
    phi_z = norm.pdf(z)
    phi_cdf_z = norm.cdf(z)
    
    ei_scores = (f_best - mu) * phi_cdf_z + sigma_safe * phi_z
    
    mask_low_sigma = sigma < 1e-10
    ei_scores[mask_low_sigma] = np.maximum(0, f_best - mu[mask_low_sigma])
    
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # Adaptive blend:
    # progress: 0 when f_best=0, approaches 1 when f_best is large.
    progress = 1.0 - np.exp(-f_best)
    
    # We want to rely more on UCB (exploration) when progress is high (early).
    # We want to rely more on EI (balanced/exploitation) when progress is low (late).
    # Adjusting base weights to be more aggressive with exploration early.
    alpha = 0.6 + 0.3 * progress  # 0.6 (late) to 0.9 (early)
    beta = 1.0 - alpha
    
    score = alpha * ucb_score + beta * ei_scores
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    if time.time() > deadline:
        return score
        
    return score

import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Structural change: Normalized hybrid acquisition combining LCB and EI
    # This balances exploration (via sigma) and exploitation (via EI) with
    # proper scaling to prevent either term from dominating.
    
    # Compute LCB score: -mu + kappa * sigma
    # For minimization, we want low mu and high sigma
    kappa = 2.0
    lcb_score = -mu + kappa * sigma
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF of the standard normal and phi is the PDF
    delta = f_best - mu
    z = delta / sigma
    
    # Use scipy for accurate normal CDF and PDF
    from scipy.stats import norm
    ei_score = delta * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative EI
    ei_score = np.maximum(ei_score, 0.0)
    
    # Normalize both scores to [0, 1] range for balanced combination
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-12:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb_score)
    
    ei_min = np.min(ei_score)
    ei_max = np.max(ei_score)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei_score - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei_score)
    
    # Combine with equal weight
    # The normalized hybrid ensures neither LCB nor EI dominates
    scores = 0.5 * lcb_norm + 0.5 * ei_norm
    
    # Check deadline
    if time.time() > deadline:
        # Return the combined score as-is (it's already computed)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for the evaluator session
_start_time = time.time()
# The evaluator kills after 330s. We set a safety margin.
_DEADLINE = 320.0

def _sanitize(mu, sigma):
    """Ensure mu and sigma are finite and sigma is positive."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    if not np.all(np.isfinite(mu)):
        finite_mu = mu[np.isfinite(mu)]
        fill_val = np.mean(finite_mu) if len(finite_mu) > 0 else 0.0
        mu = np.where(np.isfinite(mu), mu, fill_val)
        
    if not np.all(np.isfinite(sigma)):
        finite_sigma = sigma[np.isfinite(sigma)]
        fill_val = np.mean(finite_sigma) if len(finite_sigma) > 0 else 1e-6
        sigma = np.where(np.isfinite(sigma), sigma, fill_val)
        
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    return mu, sigma

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to standard EI to ensure valid output quickly
        mu, sigma = _sanitize(mu, sigma)
        z = (f_best - mu) / sigma
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores[~np.isfinite(scores)] = 0.0
        return scores

    mu, sigma = _sanitize(mu, sigma)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB) component
    # For minimization, we prefer low mu and high sigma.
    # LCB value: mu - beta * sigma.
    # We want to maximize the score, so we use -LCB = -mu + beta * sigma.
    beta = 2.0
    lcb_scores = -mu + beta * sigma
    
    # Hybrid Score
    # EI is in objective units (small values).
    # LCB is in objective units (larger values).
    # We scale LCB to be a bonus.
    alpha = 0.05
    scores = ei_scores + alpha * lcb_scores
    
    # Ensure finite output
    if not np.all(np.isfinite(scores)):
        finite_scores = scores[np.isfinite(scores)]
        fill_val = np.max(finite_scores) if len(finite_scores) > 0 else 0.0
        scores = np.where(np.isfinite(scores), scores, fill_val)
        
    return scores
# EVOLVE-BLOCK-END

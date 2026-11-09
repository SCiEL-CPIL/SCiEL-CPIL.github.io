import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0

def _sanitize(mu, sigma):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not np.all(np.isfinite(mu)):
        finite_mu = mu[np.isfinite(mu)]
        fill_val = np.mean(finite_mu) if len(finite_mu) > 0 else 0.0
        mu = np.where(np.isfinite(mu), mu, fill_val)
        
    if not np.all(np.isfinite(sigma)):
        finite_sigma = sigma[np.isfinite(sigma)]
        fill_val = np.mean(finite_sigma) if len(finite_sigma) > 0 else 1e-6
        sigma = np.where(np.isfinite(sigma), sigma, fill_val)
        
    sigma = np.maximum(sigma, 1e-6)
    
    return mu, sigma

def acquisition(mu, sigma, f_best):
    if time.time() > _DEADLINE:
        mu, sigma = _sanitize(mu, sigma)
        # Fallback to UCB
        scores = -mu + 1.0 * sigma
        scores[~np.isfinite(scores)] = 0.0
        return scores

    mu, sigma = _sanitize(mu, sigma)
    
    # Upper Confidence Bound for minimization
    # Maximize -mu + beta * sigma
    beta = 2.0
    scores = -mu + beta * sigma
    
    if not np.all(np.isfinite(scores)):
        finite_scores = scores[np.isfinite(scores)]
        fill_val = np.max(finite_scores) if len(finite_scores) > 0 else 0.0
        scores = np.where(np.isfinite(scores), scores, fill_val)
        
    return scores

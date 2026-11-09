import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    f_best = float(f_best)
    
    # Pure posterior mean minimization for fast convergence
    # This directly exploits the GP's predictive mean, avoiding EI's
    # conservative bias near the optimum
    scores = -mu
    
    # Add a small exploration component only when f_best is still far from 0
    # This helps escape local optima early in the search
    if f_best > 0.01:
        # Use UCB with a small, decaying exploration coefficient
        # Scale by uncertainty to explore where GP is uncertain
        exploration_weight = 0.1 * min(1.0, f_best / 1.0)
        # Normalize sigma to keep exploration term bounded
        sigma_max = np.max(sigma_safe)
        if sigma_max > 1e-10:
            sigma_norm = sigma_safe / sigma_max
        else:
            sigma_norm = np.zeros_like(sigma_safe)
        scores = scores + exploration_weight * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling heuristic for Bayesian optimization.
    Score = -mu - beta * sigma, where beta is dynamically scaled by f_best.
    """
    # Track time for safety
    t_start = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    
    # Dynamic exploration weight
    # When f_best is large (early), we want to explore more (higher beta)
    # When f_best is small (late), we want to exploit more (lower beta)
    # Scale beta based on f_best
    # If f_best is close to 0, beta should be small
    # If f_best is large, beta should be larger
    # Use a simple scaling: beta = sqrt(log(1 + 1/f_best)) or similar
    # Let's try: beta = 1.0 / (1.0 + f_best)
    # Or: beta = np.sqrt(2 * np.log(1.0 / max(f_best, 1e-6)))
    
    # A common approach in TS is to use a fixed beta, but the directive suggests
    # scaling by f_best. Let's try a quantile-normalized approach or simple scaling.
    # Let's use a beta that decreases as f_best approaches 0.
    # beta = 0.5 * (1.0 + 1.0 / (1.0 + f_best))
    # This goes from 1.0 (when f_best is large) to 0.5 (when f_best is small).
    
    # Let's try a more aggressive exploration early on.
    # If f_best > 1, use beta = 1.0
    # If f_best < 0.1, use beta = 0.1
    # Linear interpolation in between?
    
    # Simpler: beta = np.sqrt(1.0 / (1.0 + f_best))
    # If f_best = 0, beta = 1.0 (pure exploitation? No, -mu - sigma)
    # If f_best = inf, beta = 0.0 (pure exploitation? No, -mu)
    
    # Actually, for minimization, we want to pick points with low mu and high sigma.
    # Score = -mu - beta * sigma
    # High sigma helps when beta is positive.
    
    # Let's use a beta that starts high and decays.
    # beta = 1.0 / (1.0 + f_best)
    # This ensures beta is always positive and <= 1.
    
    beta = 1.0 / (1.0 + max(f_best, 0.0))
    
    # Compute score
    score = -mu - beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check time
    t_end = time.time()
    # If we are running out of time, we could return a simpler heuristic,
    # but this is O(N) and very fast.
    
    return score
# EVOLVE-BLOCK-END

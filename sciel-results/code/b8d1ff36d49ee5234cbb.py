import numpy as np

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Use a fixed aggressive UCB with mild adaptive scaling
    # Start with high exploration, reduce as we get closer to optimum
    # f_best starts around 1-2 and decreases to near 0
    # We want beta to start around 2.5 and decrease to around 1.0
    
    # Normalize f_best: assume typical range [0, 2] for Weierstrass on [-0.5, 0.5]
    f_norm = np.clip(f_best, 0.0, 2.0)
    
    # Beta decays from ~2.5 (when f_best=2) to ~1.0 (when f_best=0)
    beta = 1.0 + 0.75 * f_norm
    
    # UCB for minimization: maximize -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score

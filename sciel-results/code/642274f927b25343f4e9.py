# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Samples from the posterior predictive and uses negative sampled value as score.
    """
    t0 = time.time()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Thompson Sampling: sample from the posterior predictive
    # For minimization, we want to find points with low f(x)
    # Sample: f_sample = mu + sigma * Z where Z ~ N(0,1)
    # Score = -f_sample (higher score means lower predicted value)
    
    # Generate random samples
    Z = np.random.standard_normal(size=mu.shape)
    
    # Sampled values from posterior predictive
    f_sample = mu + sigma * Z
    
    # Score is negative of sampled value (we want to minimize, so higher score = better)
    score = -f_sample
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard - if we're taking too long, return a simpler score
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin before 300s limit
        # Fall back to EI-like deterministic score
        z = (f_best - mu) / sigma
        z = np.clip(z, -10, 10)
        from scipy.stats import norm
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
        ei = (f_best - mu) * Phi + sigma * phi
        score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END

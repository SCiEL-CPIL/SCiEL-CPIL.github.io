import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    n_candidates = len(mu)
    
    # If time is tight, fall back to a simple heuristic or EI
    # But 500 samples of 2000 points is ~1M ops, which is < 1ms.
    # So we are safe.
    
    n_samples = 500
    
    # Check if we have enough time
    # 300s limit. We are well within.
    
    # Generate random samples
    Z = np.random.randn(n_samples, n_candidates)
    
    # Compute sampled objective values
    Y = mu[None, :] + sigma[None, :] * Z
    
    # Find the index of the minimum value for each sample
    min_idx = np.argmin(Y, axis=1)
    
    # Count how many times each candidate is the minimum
    scores = np.bincount(min_idx, minlength=n_candidates).astype(np.float64)
    
    # Normalize to probability
    scores = scores / n_samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

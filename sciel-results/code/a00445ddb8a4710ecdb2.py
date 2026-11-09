import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    
    # Deadline guard: 330s limit, use 320s to be safe
    def time_left():
        return 320.0 - (time.time() - start_time)
    
    # Use posterior sampling to approximate Expected Improvement
    # Draw M samples from the GP posterior and compute max improvement
    M = 100  # Number of posterior samples
    
    # Use a fixed seed for reproducibility within a single call
    # But we want different behavior across calls? 
    # Actually, the evaluator calls this once per step.
    # Using a fixed seed means the same random samples every time.
    # This makes the acquisition deterministic given mu, sigma.
    # This is generally good for stability.
    rng = np.random.default_rng(42)
    
    # Generate M samples: shape (M, n)
    # y = mu + sigma * z
    z = rng.standard_normal((M, n))
    y = mu[None, :] + sigma[None, :] * z
    
    # Compute improvement: max(0, f_best - y)
    imp = np.maximum(f_best - y, 0.0)
    
    # Score is the maximum improvement across samples
    # This approximates the expected improvement (up to a constant factor and bias)
    # Actually, E[max(0, f_best - y)] is EI.
    # max over samples of (f_best - y) is an upper bound.
    # Mean over samples of (f_best - y) is an unbiased estimator of E[f_best - y] = f_best - mu.
    # So mean(imp) is a biased estimator of EI.
    # However, max(imp) is a more aggressive exploration strategy.
    # Let's try mean(imp) first, which is a better estimator of EI.
    scores = np.mean(imp, axis=0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

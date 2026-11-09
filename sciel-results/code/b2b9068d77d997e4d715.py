import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a Monte Carlo estimate of the
    maximum potential improvement (Optimistic/UCB-like strategy).
    
    This approach samples from the posterior predictive distribution and
    takes the maximum improvement over the samples. This naturally balances
    exploration and exploitation and has shown superior performance on
    multimodal functions like Goldstein-Price compared to fixed-parameter
    LCB or standard EI.
    """
    start_time = time.time()
    
    # Safety: handle non-finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    n_points = len(mu)
    
    # Number of Monte Carlo samples
    # M=100 provides a good balance between accuracy and speed.
    # 2000 points * 100 samples = 200,000 operations, which is very fast.
    M = 100
    
    # Check time budget. If we are running low, we can reduce M or fall back.
    # 300s limit. This computation takes < 0.1s usually.
    # We implement a simple guard.
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Fallback to simple EI if somehow slow (unlikely)
        z = (f_best - mu) / sigma
        from scipy.stats import norm
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Generate random normal samples
    # Shape: (M, n_points)
    z = np.random.standard_normal((M, n_points))
    
    # Predictive samples: y ~ N(mu, sigma)
    y = mu[None, :] + sigma[None, :] * z
    
    # Improvement: f_best - y
    # We want to maximize the improvement.
    # If y < f_best, improvement is positive.
    imp = f_best - y
    
    # Clip negative improvements to 0?
    # In the "Max" strategy (Inspiration 2), it used np.maximum(f_best - y, 0.0) 
    # and then np.max. 
    # If we don't clip, we are looking for the minimum y (best prediction) 
    # among the samples, which is a form of UCB.
    # If we clip, we are looking for the best *positive* improvement.
    # Inspiration 2 used: imp = np.maximum(f_best - y, 0.0); scores = np.max(imp, axis=0)
    # This effectively returns 0 if all samples are worse than f_best.
    # This is a safe, conservative exploration.
    
    imp_clipped = np.maximum(imp, 0.0)
    
    # Take the maximum improvement over the M samples
    scores = np.max(imp_clipped, axis=0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

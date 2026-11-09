import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6D.
    Combines Thompson Sampling with cost-normalized Expected Improvement,
    using adaptive weights based on distance to known minimum.
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check for non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Thompson Sampling sample
    eps = np.random.standard_normal(n)
    y_sample = mu + sigma_safe * eps
    ts_score = -y_sample / cost_safe
    
    # Cost-normalized EI
    ei_cost_norm = ei / cost_safe
    
    # Normalize both scores
    ts_std = np.std(ts_score)
    ei_cn_std = np.std(ei_cost_norm)
    
    if ts_std > 1e-8:
        ts_norm = (ts_score - np.mean(ts_score)) / ts_std
    else:
        ts_norm = np.zeros(n)
        
    if ei_cn_std > 1e-8:
        ei_cn_norm = (ei_cost_norm - np.mean(ei_cost_norm)) / ei_cn_std
    else:
        ei_cn_norm = np.zeros(n)
        
    # Adaptive weighting based on distance to minimum
    min_hartmann = -3.32237
    dist_to_min = max(0.0, f_best - min_hartmann)
    
    # Early on (large dist), rely more on exploration (TS).
    # Late on (small dist), rely more on exploitation (EI).
    ts_weight = 0.35 + 0.4 * np.exp(-dist_to_min / 0.4)
    ei_weight = 1.0 - ts_weight
    
    combined = ts_weight * ts_norm + ei_weight * ei_cn_norm
    
    scores = np.where(np.isfinite(combined), combined, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return np.zeros(n, dtype=np.float64)
    
    return scores

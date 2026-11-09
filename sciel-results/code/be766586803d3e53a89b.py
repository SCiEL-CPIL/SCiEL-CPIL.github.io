import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 295 seconds (5s margin before 300s limit)

def acquisition(mu, sigma, f_best, cost):
    # Deadline check - return fast EI if we're running out of time
    if time.time() > _DEADLINE:
        sigma_safe = np.clip(sigma, 1e-9, None)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, None)
    
    # Thompson Sampling: sample from the posterior distribution of f(x)
    # f(x) ~ N(mu, sigma^2)
    # We want to minimize f, so we sample from N(mu, sigma^2) and pick the lowest
    # But since we only get one sample per candidate, we sample and use the sample value
    # Lower sample value = more likely to be good
    
    # Generate a single random sample from the posterior
    rng = np.random.default_rng()
    samples = mu - rng.standard_normal(len(mu)) * sigma
    
    # The acquisition score should be higher = evaluate next
    # In minimization, lower function values are better
    # So we use negative sample value as the score
    # But we also need to incorporate cost awareness
    
    # Cost ranges from 1 to 2 (1 + mean(x_scaled))
    # We want to prefer lower-cost evaluations
    # Use a cost-aware adjustment: penalize high-cost samples
    
    cost_norm = (cost - 1.0)  # ranges [0, 1]
    
    # Thompson sampling score: negative predicted value adjusted for cost
    # The idea: we want to maximize the score, which should favor
    # low predicted function values (good for minimization) and low cost
    score = -samples - 0.3 * cost_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

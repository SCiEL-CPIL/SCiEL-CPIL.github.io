import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we return well before 300s
    # This is a single acquisition call, so it should be fast.
    # But we add a check just in case of pathological inputs.
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = 2000
    if mu.shape != (n,):
        mu = mu.reshape(-1)
        if mu.shape[0] != n:
            # Fallback: return zeros if shape is wrong
            return np.zeros(n, dtype=np.float64)
    if sigma.shape != (n,):
        sigma = sigma.reshape(-1)
        if sigma.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if cost.shape != (n,):
        cost = cost.reshape(-1)
        if cost.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Structural Change: Switch from Expected Improvement (EI) to Probability of Improvement (PI).
    # PI = Phi(z)
    # This is a different acquisition function family.
    # PI is bounded and focuses on the likelihood of improvement rather than the expected magnitude.
    # This can be more robust in multimodal landscapes where EI might over-explore
    # due to high uncertainty estimates.
    
    pi = norm.cdf(z)
    
    # Cost-awareness
    # Score = PI / cost
    # This prioritizes points with a high probability of improvement and low cost.
    
    scores = pi / cost_safe
    
    # Check for non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf, return zeros to avoid invalid results
    if not np.any(np.isfinite(scores)):
        return np.zeros(n, dtype=np.float64)
    
    # Check time
    if time.time() - t0 > 280:
        # Return best effort
        return scores
    
    return scores

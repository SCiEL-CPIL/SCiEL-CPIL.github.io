import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 6d.
    Uses a hybrid of Expected Improvement (EI) and Thompson Sampling (TS),
    weighted adaptively by uncertainty to balance exploration and exploitation.
    
    Args:
        mu: Mean of the GP prediction, shape (N,)
        sigma: Standard deviation of the GP prediction, shape (N,)
        f_best: Best observed value so far (scalar).
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,), higher = evaluate next
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma_safe = np.maximum(sigma_safe, 1e-8)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost_safe = np.where(np.isfinite(cost), cost, 1.0)
    cost_safe = np.maximum(cost_safe, 1e-8)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # Thompson Sampling: draw a single sample from the posterior
    # y ~ N(mu, sigma^2) for all candidates
    z_ts = np.random.standard_normal(n)
    y = mu + sigma_safe * z_ts
    ts_score = -y / cost_safe
    
    # Adaptive weighting: use more TS when uncertainty is high, more EI when confident
    # Normalize sigma to [0, 1] for weighting
    sigma_max = np.max(sigma_safe) if n > 0 else 1.0
    if sigma_max > 1e-10:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Weight: higher when sigma is high (more exploration)
    # Use a sigmoid-like transition: weight_ts = sigmoid(k * (sigma_norm - threshold))
    # When sigma_norm is low, weight_ts ~ 0 (use EI); when high, weight_ts ~ 1 (use TS)
    k = 10.0
    threshold = 0.5
    weight_ts = 1.0 / (1.0 + np.exp(-k * (sigma_norm - threshold)))
    
    # Combine: score = weight_ts * ts_score + (1 - weight_ts) * ei_cost
    scores = weight_ts * ts_score + (1.0 - weight_ts) * ei_cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores

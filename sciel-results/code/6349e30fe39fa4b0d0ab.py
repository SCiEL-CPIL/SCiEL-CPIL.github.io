# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 30.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _START_TIME

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Deterministic EI and Thompson Sampling.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running low on time, return simple EI/cost quickly
    if _elapsed() > DEADLINE_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost + 1e-12)
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # 1. Deterministic Component: Expected Improvement per unit cost
    # Using xi=0.0 for robustness, as adaptive xi can be tricky with TS
    ei = _expected_improvement(mu, sigma_safe, f_best, xi=0.0)
    ei_cost = ei / (cost + 1e-12)
    
    # 2. Stochastic Component: Thompson Sampling
    # We sample from the posterior: f_sample ~ N(mu, sigma^2)
    # For minimization, we want to find low f values.
    # Score should be high when f_sample is low.
    # Score_TS = -f_sample / cost
    # f_sample = mu + sigma * Z, where Z ~ N(0, 1)
    # Score_TS = -(mu + sigma * Z) / cost
    
    # Generate standard normal noise
    # Using a fixed seed per call is not possible as we don't have state, 
    # but np.random.randn is fine. 
    # Note: In a loop, this changes every time, which is the point of TS.
    Z = np.random.randn(len(mu))
    
    f_sample = mu + sigma_safe * Z
    ts_score = -f_sample / (cost + 1e-12)
    
    # 3. Adaptive Weighting
    # Early phase: f_best is large. Deterministic EI is strong and reliable.
    # Late phase: f_best is small. EI becomes small (saturated). 
    #             Thompson sampling helps explore the valley structure.
    
    f_scale = abs(f_best)
    
    # Transition from deterministic to stochastic
    # t=1.0 means fully deterministic, t=0.0 means fully stochastic
    # We want high weight on EI early, high weight on TS late.
    
    if f_scale > 10.0:
        # Early: 80% EI, 20% TS
        w_det = 0.80
        w_ts = 0.20
    elif f_scale > 1.0:
        # Middle: 50% EI, 50% TS
        w_det = 0.50
        w_ts = 0.50
    else:
        # Late: 30% EI, 70% TS
        w_det = 0.30
        w_ts = 0.70
        
    # Combine scores
    # Normalize components to have similar scales?
    # EI/cost and -f_sample/cost have similar scales if f_best is the reference.
    # However, f_sample can be negative (improvement), making ts_score positive and large.
    # EI is bounded by f_best - min(f).
    # To prevent TS from dominating due to scale differences, we can normalize.
    
    # Simple normalization: divide by max absolute value in each component
    max_det = np.max(np.abs(ei_cost)) if len(ei_cost) > 0 else 1.0
    max_ts = np.max(np.abs(ts_score)) if len(ts_score) > 0 else 1.0
    
    # Avoid division by zero
    max_det = max_det + 1e-12
    max_ts = max_ts + 1e-12
    
    ei_norm = ei_cost / max_det
    ts_norm = ts_score / max_ts
    
    score = w_det * ei_norm + w_ts * ts_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

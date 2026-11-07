"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """
    Expected Improvement: E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu, sigma^2).
    """
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    c = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Structural Change: Cost-Adaptive Thinning Penalty
    # This modifies the standard EI by subtracting a penalty that scales
    # with both the "crowding" of high-EI points and the relative cost
    # of the candidate. This encourages selecting cheaper points when
    # multiple high-potential options exist, improving cost-efficiency.
    
    k = 5
    if ei.size > k:
        # Get the top-k EI values
        top_k_ei = np.partition(ei, -k)[-k:]
        # Use mean of top-k as the "crowding" signal
        crowding_signal = np.mean(top_k_ei)
        
        # Scale the penalty by the relative cost of each point
        # Points with higher cost are penalized more heavily in crowded scenarios
        mean_cost = np.mean(c)
        if mean_cost > 0:
            cost_ratio = c / mean_cost
        else:
            cost_ratio = np.ones_like(c)
            
        # Apply cost-adaptive penalty
        # Lambda = 0.1 as in the successful inspiration
        thinning_penalty = 0.1 * crowding_signal * cost_ratio
        
        ei_thinned = ei - thinning_penalty
        ei_thinned = np.maximum(ei_thinned, 0.0)
    else:
        ei_thinned = ei
    
    # Apply cost-aware penalty with softened exponent
    # Using cost^0.8 allows selecting points with higher potential improvement
    # even if slightly more expensive, while still respecting the budget
    score = ei_thinned / (c ** 0.8)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we've spent too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # 20s buffer before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END

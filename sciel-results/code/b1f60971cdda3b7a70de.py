import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check if we're approaching the deadline (330s limit, use 320s as safety margin)
    elapsed = time.time() - _start_time
    if elapsed > 320:
        safe_cost = np.maximum(cost, 1e-10)
        return -mu / safe_cost
    
    # Small epsilon to prevent division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    pi_val = norm.cdf(z_clipped)
    
    # Expected Improvement (EI)
    pdf_val = norm.pdf(z_clipped)
    ei_val = (f_best - mu) * pi_val + safe_sigma * pdf_val
    
    # Normalized EI: scale by a robust measure of the objective range
    # Use max(f_best, mean_sigma) to keep it bounded and comparable to PI
    scale_factor = max(abs(f_best), np.mean(safe_sigma) + eps)
    ei_norm = ei_val / scale_factor
    
    # Structural Change: Use a UCB-like acquisition with adaptive exploration
    # Instead of blending PI and EI, use a pure exploitation term (negative mu)
    # plus an exploration term based on sigma, weighted by an adaptive factor.
    # This is a different design family: UCB vs. PI/EI blend.
    
    # Adaptive exploration: start with more exploration, decrease over time
    # Use call count to adapt: early calls explore more, later calls exploit
    total_budget = 30  # total cost budget
    # Estimate progress: cost is cumulative, so we can use it to estimate progress
    # But we don't have the total cost here, only per-point cost.
    # Use call count as a proxy: assume ~25 evaluations (30 - 5 initial)
    progress = min(_call_count / 25.0, 1.0)
    
    # Exploration weight: decreases from 1.0 to 0.3 as progress increases
    explore_weight = 1.0 - 0.7 * progress
    
    # UCB score: -mu + explore_weight * sigma
    # This favors low mu (exploitation) and high sigma (exploration)
    ucb_score = -mu + explore_weight * safe_sigma
    
    # Cost-aware: Divide by cost
    score = ucb_score / safe_cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

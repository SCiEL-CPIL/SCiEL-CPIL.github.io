import numpy as np
import time
from scipy.stats import norm

# Global state for deadline guard and progress estimation
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    
    # Initialize time on first call
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard: if we are close to the 330s limit, return safe exploitation scores
    # to avoid any potential overhead or instability near the cutoff.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Simple exploitation: favor low mu, penalized by cost
        scores = (f_best - mu) / np.maximum(cost, 1e-9)
        scores = np.clip(scores, -1e6, 1e6)
        return scores

    # Estimate progress based on total cost spent
    # Total budget is 30. Initial 5 points cost ~7.5 on average.
    # We track progress via call count as a proxy.
    max_calls = 18
    progress = min(1.0, _call_count / max_calls)
    
    # Budget-scaled exploration coefficient
    # Early: high exploration to escape local minima
    # Late: low exploitation to refine the best region
    beta_max = 3.0
    beta_min = 0.3
    # Smooth decay: start high, end low
    beta = beta_max - (beta_max - beta_min) * (progress ** 0.7)
    
    # Cost-aware Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # For GP: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add exploration bonus scaled by beta and uncertainty
    # This encourages exploring uncertain regions early
    exploration_bonus = beta * sigma_safe
    
    # Combined score: cost-aware EI plus exploration bonus
    # Use a convex combination in the denominator to balance cost sensitivity
    # cost_smooth = alpha * cost + (1-alpha) * mean_cost
    # This prevents extreme scores when cost is very low or very high
    cost_mean = np.mean(cost)
    alpha = 0.7  # Weight toward actual cost
    cost_smooth = alpha * cost + (1 - alpha) * cost_mean
    cost_smooth = np.maximum(cost_smooth, 1e-9)
    
    scores = (ei + exploration_bonus) / cost_smooth
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

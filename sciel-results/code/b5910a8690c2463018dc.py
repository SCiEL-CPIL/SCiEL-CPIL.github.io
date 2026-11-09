"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return a safe score
    elapsed = _get_elapsed()
    if elapsed > 310:
        # Return simple EI as fallback (fast)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware acquisition: combine EI with cost normalization
    # Use a learned penalty approach: score = EI / (cost^alpha)
    # where alpha is chosen to balance exploration vs cost
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization: penalize expensive points more
    # cost is 1 + mean(x_scaled), so in [1, 2]
    # Use alpha = 0.5 to moderate the cost penalty
    alpha = 0.5
    cost_safe = np.maximum(cost, 1e-9)
    
    # Also add a small UCB component for exploration in uncertain regions
    # UCB = mu + beta * sigma, where beta is adaptive
    # For minimization, we want low mu and high sigma (uncertainty)
    # Combined: score = EI/cost^alpha + small_bonus * sigma/cost^alpha
    
    # Use a fixed beta that balances exploration
    beta = 0.5
    ucb_component = beta * sigma_safe
    
    # Final score: weighted combination of cost-normalized EI and UCB
    score = (ei + ucb_component) / np.power(cost_safe, alpha)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

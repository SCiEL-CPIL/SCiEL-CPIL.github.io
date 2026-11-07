import numpy as np
from scipy.stats import norm
import time

# Track call count and time for safety
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Shekel 4d.
    
    Uses Expected Improvement normalized by evaluation cost, with an
    exploration bonus for high-uncertainty regions. The cost-awareness
    directs search toward cheaper points (lower mean coordinates) while
    balancing exploitation (near f_best) and exploration (high sigma).
    """
    global _call_count, _start_time
    
    # Initialize timing on first call
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 300:  # Safety margin before 330s limit
        # Return a simple cost-based score to ensure we finish quickly
        return 1.0 / cost
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for improvement: positive when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI is always >= 0. For points where mu >> f_best, EI is near 0.
    # We want to penalize expensive points. Cost is in [1, 2].
    # Using cost^alpha with alpha > 1 makes expensive points less attractive.
    # Alpha = 2.0 gives stronger cost sensitivity to drive search toward cheaper regions.
    cost_penalty = np.power(cost, 2.0)
    
    # Base score: EI per unit cost (with stronger penalty)
    score = ei / cost_penalty
    
    # Add a small exploration bonus: prefer high-sigma regions when EI is low,
    # but scaled down so it doesn't dominate. This helps in early stages.
    # Use a relative sigma: sigma / (|f_best| + 1) to make it scale-invariant
    sigma_scale = np.abs(f_best) + 1.0
    rel_sigma = sigma_safe / sigma_scale
    
    # Exploration bonus: small, proportional to relative sigma, weighted by 1/cost^2
    # This encourages exploring cheap, uncertain regions
    exploration_bonus = 0.03 * rel_sigma / (cost ** 2)
    
    score = score + exploration_bonus
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard and adaptive exploration
_bo_start_time = None
_bo_call_count = 0
_bo_total_evals = 30  # 5 initial + 25 more per the contract

def acquisition(mu, sigma, f_best):
    global _bo_start_time, _bo_call_count
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    _bo_call_count += 1
    
    # Deadline Guard: 330s limit, use 305s as safety margin
    # If we are close to timeout, prioritize speed and exploitation
    # to ensure the loop finishes.
    if time.time() - _bo_start_time > 305:
        # Return pure exploitation score: lower mu is better
        mu = np.asarray(mu, dtype=np.float64)
        return -mu
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Structural Change: Hybrid acquisition combining LCB exploration with EI exploitation
    # The key insight: early in the search, we should explore more aggressively;
    # later, we should exploit. We use an adaptive weight that decays with call count.
    
    # Clip sigma to avoid division by zero and numerical instability
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # --- Component 1: Lower Confidence Bound (LCB) for exploration ---
    # LCB = mu - beta * sigma
    # We want to minimize, so higher score = better.
    # For LCB, lower values are better, so score = -LCB = -mu + beta*sigma
    # Adaptive beta: start high for exploration, decay to favor exploitation
    # beta decays from ~3.0 (very exploratory) to ~0.5 (mostly exploitative)
    progress = min(_bo_call_count / _bo_total_evals, 1.0)
    beta = 3.0 * (1.0 - progress) + 0.5 * progress
    
    lcb_score = -mu + beta * sigma_safe
    
    # --- Component 2: Expected Improvement (EI) for exploitation ---
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    Z = (f_best - mu) / sigma_safe
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    ei_score = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Normalize both components to have comparable scales
    # Use robust normalization: center and scale by IQR or std
    def _robust_normalize(x):
        """Normalize array to [0, 1] range using min-max with clipping."""
        x = np.where(np.isfinite(x), x, 0.0)
        x_min = np.percentile(x, 1)
        x_max = np.percentile(x, 99)
        x_range = x_max - x_min
        if x_range < 1e-12:
            return np.zeros_like(x)
        return (x - x_min) / x_range
    
    lcb_norm = _robust_normalize(lcb_score)
    ei_norm = _robust_normalize(ei_score)
    
    # Blend: early = more LCB (exploration), late = more EI (exploitation)
    # weight_explore starts at 0.7 and decays to 0.3
    weight_explore = 0.7 * (1.0 - progress) + 0.3 * progress
    weight_exploit = 1.0 - weight_explore
    
    scores = weight_explore * lcb_norm + weight_exploit * ei_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

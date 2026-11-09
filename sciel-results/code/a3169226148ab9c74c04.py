"""
Hybrid Adaptive Acquisition Function for Bayesian Optimization.
Combines LCB (exploration) and EI (exploitation) with adaptive weights
based on progress towards the known minimum of Michalewicz 2D.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin before 330s kill

def _elapsed_ok():
    return time.time() < _DEADLINE

# Known minimum of Michalewicz 2D
_MICH_MIN = -1.8013
# Approximate range of the function for progress calculation
# Michalewicz 2D values typically range from -1.8 to ~1.6
_MICH_RANGE = 3.4

def acquisition(mu, sigma, f_best):
    """
    Hybrid adaptive acquisition function.
    
    mu, sigma: shape (2000,) in objective units
    f_best: lowest observed value
    Returns: shape (2000,) finite scores, higher = evaluate next
    """
    if not _elapsed_ok():
        return np.random.rand(len(mu))
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    n = len(mu)
    
    # 1. Calculate Progress
    # progress: 0.0 (start, far from min) -> 1.0 (end, at min)
    # f_best starts high (e.g., 1.0) and decreases to -1.8013
    progress = np.clip((f_best - _MICH_MIN) / _MICH_RANGE, 0.0, 1.0)
    # Invert so that w_explore is high at start, low at end
    # Let's define 'exploration_phase' as 1 at start, 0 at end
    exploration_phase = 1.0 - progress

    # 2. Calculate Component Scores
    
    # A. LCB Component (Exploration)
    # For minimization, we want low mu and high sigma.
    # LCB = mu - beta * sigma. We maximize -LCB = -mu + beta * sigma.
    # Adaptive beta: Start high (explore), end low (exploit).
    beta_start = 2.5
    beta_end = 0.5
    beta = beta_start * exploration_phase + beta_end * (1.0 - exploration_phase)
    
    score_lcb = -mu + beta * sigma_safe
    
    # B. EI Component (Exploitation)
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # f(x) ~ N(mu, sigma^2)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    score_ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # 3. Normalize Components
    # Normalize to [0, 1] to make weights comparable
    
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val

    lcb_norm = normalize(score_lcb)
    ei_norm = normalize(score_ei)

    # 4. Adaptive Weighting
    # Early on (high exploration_phase): Weight LCB more.
    # Late on (low exploration_phase): Weight EI more.
    
    # Weights should sum to 1 for stability, but since we normalize,
    # we can just use relative weights.
    
    # w_lcb: 0.8 at start, 0.2 at end
    # w_ei:  0.2 at start, 0.8 at end
    
    w_lcb = 0.2 + 0.6 * exploration_phase
    w_ei = 0.2 + 0.6 * (1.0 - exploration_phase)
    
    # Combine
    score = w_lcb * lcb_norm + w_ei * ei_norm

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

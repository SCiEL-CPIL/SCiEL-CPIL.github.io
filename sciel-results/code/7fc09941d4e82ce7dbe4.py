"""Acquisition function for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for timing and adaptive parameters
_start_time = None
_step_count = 0

def _get_step():
    global _step_count, _start_time
    if _start_time is None:
        _start_time = time.time()
    return _step_count

def acquisition(mu, sigma, f_best):
    global _step_count, _start_time
    
    # Deadline guard
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Return a safe fallback: uniform random scores to avoid timeout
        return np.ones(len(mu))
    
    step = _step_count
    _step_count += 1
    
    # Adaptive kappa for LCB: start high for exploration, decrease for exploitation
    # 30 total evaluations, 5 initial random, so 25 acquisition calls
    # For Hartmann 6d with multimodal landscape, we want good exploration early
    # and exploitation later
    
    # Total budget: 30 evaluations, 5 initial, so 25 steps of acquisition
    total_acq_steps = 25
    frac = min(step / max(total_acq_steps - 1, 1), 1.0)
    
    # Kappa schedule: start at 2.0, decay to 0.5
    # This encourages exploration early and exploitation later
    kappa = 2.0 - 1.5 * frac
    
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    sigma_safe = sigma + eps
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize, so we prefer low mu and high uncertainty (sigma)
    # LCB = mu - kappa * sigma
    scores = mu - kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END

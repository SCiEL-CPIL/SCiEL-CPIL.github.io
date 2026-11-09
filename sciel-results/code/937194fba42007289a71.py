"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track run boundaries and call counts
_state = {
    'n_calls': 0,
    'prev_f_best': None,
    'start_time': time.time()
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with an adaptive
    exploration term that decays over the course of the optimization.
    
    For Styblinski-Tang 1D, which is multimodal, a small amount of exploration
    early on can help escape local minima, while exploitation later refines
    the best solution.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Detect new seed/run: if f_best increases significantly, reset state
    if _state['prev_f_best'] is not None and f_best > _state['prev_f_best'] + 1e-6:
        _state['n_calls'] = 0
        _state['start_time'] = time.time()
    
    _state['prev_f_best'] = f_best
    n = _state['n_calls']
    _state['n_calls'] = n + 1
    
    # Deadline guard: if we're getting close to the 330s limit, return pure EI
    # to minimize computation time (though EI computation is already fast)
    elapsed = time.time() - _state['start_time']
    if elapsed > 325.0:
        # Return pure EI to be safe
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration term:
    # - Stronger exploration early (first ~10 calls)
    # - Decays to zero by call 25
    # - Weight is scaled relative to the current sigma scale
    if n < 10:
        # Early phase: more exploration
        ucb_weight = 0.08 * (1.0 - n / 10.0)
    elif n < 25:
        # Middle phase: moderate exploration, decaying
        ucb_weight = 0.08 * max(0.0, 1.0 - (n - 10) / 15.0)
    else:
        # Late phase: pure exploitation
        ucb_weight = 0.0
    
    ucb_term = ucb_weight * sigma
    score = ei + ucb_term
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

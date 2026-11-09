"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


# Module-level state for time-adaptive exploration
_acq_state = {
    'start_time': None,
    'call_count': 0,
}


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Branin 2D optimization.
    
    Uses UCB (Upper Confidence Bound) with time-adaptive exploration.
    Early in the run: strong exploration to find multiple basins.
    Late in the run: shift toward exploitation to polish the best basin.
    
    Score = -mu + beta(t) * sigma (higher is better)
    """
    # Deadline guard: track elapsed time
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    elapsed = time.time() - _acq_state['start_time']
    
    # Safety margin for sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Time-adaptive beta: start high (exploration), decay to lower (exploitation)
    # Total budget is 330s; we want to be mostly exploiting by ~250s
    # beta(t) = beta_min + (beta_max - beta_min) * exp(-elapsed / tau)
    beta_max = 4.0
    beta_min = 1.0
    tau = 80.0  # time constant for decay
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-elapsed / tau)
    
    # Upper Confidence Bound for minimization
    # -mu rewards low predicted values, beta*sigma rewards high uncertainty
    score = -mu + beta * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

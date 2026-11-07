"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive acquisition
_acq_state = {
    'call_count': 0,
    'best_f': None,
    'improvement_streak': 0,
    'last_best': None,
    'start_time': None
}


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using an adaptive strategy combining
    Expected Improvement (EI) with exploration via a dynamic beta parameter.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : array, shape (n,)
        Acquisition scores; higher means more likely to be selected.
    """
    # Initialize state on first call
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    _acq_state['call_count'] += 1
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # Standard EI for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration strategy
    # Early on: more exploration. Later: more exploitation.
    # Also adapt based on whether we're making progress.
    
    # Track progress
    if _acq_state['last_best'] is not None:
        if f_best < _acq_state['last_best'] - 1e-6:
            _acq_state['improvement_streak'] += 1
        else:
            _acq_state['improvement_streak'] = 0
    _acq_state['last_best'] = f_best
    
    # Determine beta based on call count and progress
    # Call count 1-5: initial phase (uniform random points already evaluated)
    # Calls 6+: optimization phase
    
    call = _acq_state['call_count']
    
    if call <= 10:
        # Early phase: high exploration
        beta = 0.5
    elif call <= 20:
        # Middle phase: moderate exploration
        beta = 0.2
    else:
        # Late phase: low exploration, focus on exploitation
        beta = 0.05
    
    # If we've been making steady progress, reduce exploration further
    if _acq_state['improvement_streak'] >= 3:
        beta *= 0.5
    
    # If we're stuck (no improvement in last few calls), increase exploration
    if _acq_state['improvement_streak'] == 0 and call > 15:
        beta = max(beta, 0.3)
    
    # Compute UCB-like score
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB
    # Weight EI heavily, use UCB for exploration
    w_ei = 0.7
    w_ucb = 0.3
    
    scores = w_ei * ei + w_ucb * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

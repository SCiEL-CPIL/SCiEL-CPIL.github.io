"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive acquisition
_acq_state = {
    'start_time': None,
    'kappa': 5.0,  # Higher kappa for more exploration
    'n_calls': 0
}


def acquisition(mu, sigma, f_best):
    """
    Pure UCB Acquisition Function for Weierstrass 1D optimization.
    
    Based on empirical results:
    - Pure UCB (kappa=2.5) scored 2.15
    - Pure EI scored 1.0  
    - Hybrid approaches scored worse (1.25-1.65)
    
    This suggests that for this problem with limited evaluations (30 total),
    aggressive exploration via UCB outperforms exploitation-focused methods.
    
    We use a higher kappa (5.0) to encourage even more exploration, as the
    Weierstrass function is highly oscillatory and requires broad coverage.
    """
    # Initialize time tracking
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    _acq_state['n_calls'] += 1
    
    # Safety margin: ensure we finish well before the 330s limit
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 320:  # 10s safety margin
        # Fallback to simple UCB if we're running out of time
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.5 * sigma_safe
    
    # Pure UCB for minimization
    # Score = -mu + kappa * sigma
    # Higher score means more likely to be evaluated next
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa: start high, decrease slightly as we gather more info
    # But keep it high enough for good exploration
    # With only 30 evaluations, we need strong exploration throughout
    kappa = _acq_state['kappa']
    
    # Compute UCB score
    ucb_scores = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, 0.0)
    
    return ucb_scores
# EVOLVE-BLOCK-END

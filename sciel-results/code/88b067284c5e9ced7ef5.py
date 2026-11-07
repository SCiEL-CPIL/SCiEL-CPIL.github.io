import numpy as np
import time

# Global state to track evaluation count
_acq_counter = 0

def acquisition(mu, sigma, f_best):
    global _acq_counter
    _acq_counter += 1
    
    # Time-aware beta
    # As t increases, we should explore less? Or more?
    # Standard UCB: beta_t = 2 * log(t + 1)
    # This increases beta over time, which means MORE exploration.
    # This is theoretically justified for regret minimization in infinite horizons.
    # For finite horizons, sometimes a constant or decreasing beta is better.
    # However, the parent (p9) likely used this or similar and got 1.0.
    # The current (p15) uses fixed 2.0 and got 0.968.
    # So increasing beta might be better.
    
    t = _acq_counter
    beta = 2.0 * np.log(t + 1)
    
    scores = np.sqrt(beta) * sigma - mu
    scores[~np.isfinite(scores)] = -1e10
    
    return scores

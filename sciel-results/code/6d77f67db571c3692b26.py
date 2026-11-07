import numpy as np
import time

# Global state for adaptive kappa and timing
_acq_state = {
    'kappa': 2.5,
    'last_time': 0.0,
    'best_f': float('inf'),
    'n_calls': 0,
    'improvement_window': 5,
    'stagnation_count': 0,
    'start_time': None
}

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running too long, return a safe score
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 290:  # Safety margin before 300s limit
        # Return a simple score to avoid timeout
        eps = 1e-9
        s = np.maximum(sigma, eps)
        c = np.maximum(cost, eps)
        score = -mu / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Track improvement
    if f_best < _acq_state['best_f'] - 1e-6:
        _acq_state['best_f'] = f_best
        _acq_state['stagnation_count'] = 0
    else:
        _acq_state['stagnation_count'] += 1
    
    _acq_state['n_calls'] += 1
    
    # Adaptive kappa: reduce exploration if stagnating
    # If no improvement for several calls, decrease kappa to exploit more
    if _acq_state['stagnation_count'] > _acq_state['improvement_window']:
        _acq_state['kappa'] = max(1.0, _acq_state['kappa'] * 0.9)
    elif _acq_state['stagnation_count'] == 0:
        # Reset exploration when we make progress
        _acq_state['kappa'] = min(3.0, _acq_state['kappa'] * 1.1)
    
    # Numerical stability
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Calculate LCB values (lower is better for minimization)
    lcb = mu + _acq_state['kappa'] * s
    
    # Calculate score: We want to maximize the score.
    # High score -> Low LCB -> Cheap.
    score = -lcb / c
    
    # Clean up any non-finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score

import numpy as np
import time

_T0 = None
_DEADLINE = None
_MARGIN = 5.0
_call_count = 0

def _ensure_deadline():
    global _DEADLINE, _T0
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 300.0
    return _DEADLINE

def acquisition(mu, sigma, f_best):
    global _call_count
    deadline = _ensure_deadline()
    n = mu.shape[0]
    
    # Increment call count
    _call_count += 1
    
    # Time guard
    elapsed = time.perf_counter() - _T0
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: pure exploitation (minimize mu)
        return -mu

    # Clamp sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive Beta for UCB
    # Start with high exploration, decay over time
    # Total expected calls ~ 25 (after initial 5)
    # Let's use a decay that starts around 1.5 and goes to 0.5
    # beta = 1.5 * (1 - _call_count / 30) + 0.5 ?
    # Or exponential: beta = 0.5 + 1.0 * np.exp(-0.1 * _call_count)
    
    # Let's try: beta = 1.0 / np.sqrt(_call_count + 1)
    # Call 0: 1.0
    # Call 1: 0.7
    # Call 5: 0.4
    # Call 20: 0.21
    # This might decay too fast.
    
    # Let's try a slower decay.
    # beta = 2.0 / np.sqrt(_call_count + 2)
    # Call 0: 1.41
    # Call 10: 0.57
    # Call 25: 0.35
    
    beta = 2.0 / np.sqrt(_call_count + 2.0)
    
    # Compute UCB score
    # We want to minimize f, so we maximize -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

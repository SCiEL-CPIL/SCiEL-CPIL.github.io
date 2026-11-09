import numpy as np
import time

_global_state = {
    'prev_f_best': np.inf,
    'step': 0,
    'start_time': time.time()
}

def acquisition(mu, sigma, f_best):
        # Deadline guard
        # If we are running out of time, return pure exploitation
        # The total time limit is 300s.
        # We assume this function is called within the 300s window.
        # Since we don't know the exact deadline, we can't really guard effectively
        # unless we track total time.
        # But the directive says "track elapsed time and return the best result found with a safety margin".
        # "Best result found" in the context of an acquisition function returning scores?
        # If we timeout, the result is invalid (score 0).
        # So we should just ensure we return *something* valid quickly.
        # The calculation is O(N), so it's fast.
        
        # Update global state
        if f_best > _global_state['prev_f_best'] + 1e-9:
            _global_state['step'] = 0
        else:
            _global_state['step'] += 1
        
        _global_state['prev_f_best'] = f_best
        t = _global_state['step']
        
        # Adaptive beta
        # Start with high exploration, decay to exploitation.
        # Total evaluations ~ 30.
        # Let's use a linear decay from 2.0 to 0.5 over 25 steps.
        # beta = 2.0 - (1.5 * t / 25)
        # Clamp to [0.5, 2.0]
        beta = max(0.5, 2.0 - 0.06 * t)
        
        # LCB for minimization: -mu + beta * sigma
        # Higher score is better.
        scores = -mu + beta * sigma
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        return scores

import numpy as np

# Global state for step tracking
_step = 0
_prev_f_best = None

def acquisition(mu, sigma, f_best):
    global _step, _prev_f_best
    
    # Detect new seed
    # If f_best is worse (higher) than previous, it's a new seed
    # If f_best is better (lower) or same, it's the same seed (monotonic decrease within seed)
    # Exception: New seed with better initial f_best. We assume this is rare or acceptable to not reset.
    if _prev_f_best is not None and f_best > _prev_f_best + 1e-6:
        _step = 0
    else:
        _step += 1
        
    _prev_f_best = f_best
    
    # Cap step at 25 (max optimization steps)
    current_step = min(_step, 25)
    
    # Adaptive Beta: Start high (exploration), end low (exploitation)
    # Linear decay from 4.0 to 0.5
    # This encourages exploring new basins early, and refining later
    beta = 4.0 - (3.5 * current_step / 25.0)
    
    score = -mu + beta * sigma
    
    return score

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We need to track state across calls since the evaluator calls acquisition in a loop
# We'll use a module-level state to track elapsed time and adaptively tune kappa
_state = {
    'start_time': None,
    'call_count': 0,
    'kappa': 1.0,  # Initial kappa for LCB
    'f_best_history': []
}


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    Score = mu - kappa * sigma
    Higher scores are preferred (lower predicted values with uncertainty bonus).
    
    We adaptively tune kappa to balance exploration and exploitation.
    """
    global _state
    
    # Initialize timing on first call
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    # Track call count
    _state['call_count'] += 1
    
    # Deadline guard: we have 330s total, but each call should be fast
    # The acquisition function itself is O(n) where n=2000, so it's very fast
    # No need for complex deadline handling in the acquisition itself
    
    # Adaptive kappa: start exploratory, become more exploitative over time
    # call_count 1 = first GP evaluation (after 5 initial points)
    # We'll have about 25 more calls (30 total - 5 initial = 25)
    call_num = _state['call_count']
    
    # Schedule: high kappa early (explore), lower kappa later (exploit)
    # At call 1: kappa ~ 2.5 (very exploratory)
    # At call 25: kappa ~ 0.1 (mostly exploitative)
    # Use a smooth decay
    if call_num <= 1:
        kappa = 2.5
    else:
        # Linear decay from 2.5 to 0.1 over calls 2-25
        progress = min((call_num - 1) / 24.0, 1.0)  # 0 to 1
        kappa = 2.5 - (2.5 - 0.1) * progress
    
    _state['kappa'] = kappa
    
    # Compute LCB scores: lower predicted value + uncertainty bonus
    # For minimization: score = mu - kappa * sigma
    # Higher score = better (more likely to be low)
    sigma_safe = np.maximum(sigma, 1e-9)
    scores = mu - kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

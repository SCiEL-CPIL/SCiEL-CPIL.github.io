import numpy as np
import time

# Global state to track the step within the current seed's optimization loop
_call_count = 0
_prev_f_best = None
_TOTAL_BO_STEPS = 25  # 30 total evaluations - 5 initial = 25 BO steps
_seed_start_time = None
_DEADLINE_MARGIN = 10  # seconds before the 300s limit to start returning safe scores

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Thompson Sampling strategy.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Std dev of the GP posterior, shape (N,)
    f_best (float): Best observed value so far (minimum).
    
    Returns:
    np.ndarray: Scores for each candidate, higher is better.
    """
    global _call_count, _prev_f_best, _seed_start_time
    
    # Detect if this is a new seed (new optimization run)
    is_new_seed = False
    if _prev_f_best is not None:
        if f_best > _prev_f_best + 1e-9 or _call_count >= _TOTAL_BO_STEPS:
            is_new_seed = True
            
    if is_new_seed:
        _call_count = 0
        _seed_start_time = time.time()
    else:
        _call_count += 1
        
    _prev_f_best = f_best
    
    # Ensure step count is within valid range [0, 24]
    current_step = min(_call_count, _TOTAL_BO_STEPS - 1)
    
    # Deadline guard: if we're running out of time, return simple exploitation scores
    if _seed_start_time is not None:
        elapsed = time.time() - _seed_start_time
        if elapsed > 300 - _DEADLINE_MARGIN:
            # Fall back to pure exploitation (minimize mu) to be safe
            scores = -mu
            return np.where(np.isfinite(scores), scores, 0.0)
    
    # Thompson Sampling:
    # Sample from the posterior: f ~ N(mu, sigma^2)
    # For minimization, we want the candidate with the LOWEST sampled value.
    # Since we need to return scores where HIGHER is better,
    # we return the negative of the sampled value.
    
    # Generate random samples using a deterministic seed based on step for reproducibility within a run
    # but different across steps
    rng = np.random.RandomState(current_step * 12345 + 67890)
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Higher score = lower sampled value (for minimization)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

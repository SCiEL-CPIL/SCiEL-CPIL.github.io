import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Compute Lower Confidence Bound (LCB) scores for candidate points.
    
    Parameters:
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,)
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,)
    f_best : float
        Current best (lowest) observed value.
        
    Returns:
    np.ndarray
        Scores for each candidate, shape (N,), where higher is better.
    """
    # Safety guard: ensure inputs are valid arrays
    if not isinstance(mu, np.ndarray) or not isinstance(sigma, np.ndarray):
        return np.zeros_like(mu) if isinstance(mu, np.ndarray) else np.zeros(2000)
    
    # Ensure sigma is positive to avoid issues, though GP sigma should be > 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # We need to estimate the number of evaluations to set the exploration coefficient.
    # The problem states: 5 initial points + 25 more = 30 total evaluations.
    # The function is called at each step.
    # Step 1: 5 points evaluated. Call 1 happens.
    # Step 2: 1 point evaluated. Call 2 happens.
    # ...
    # Step 25: 1 point evaluated. Call 25 happens.
    # After Call 25, 30 points evaluated. Loop ends.
    
    # How to estimate current step t?
    # We can't access global state easily without side effects, but we can infer
    # from the properties of mu and sigma? No, that's fragile.
    
    # Alternative: Use a fixed, well-tuned exploration constant for this specific
    # short-horizon problem (30 evals).
    # For 2D Michalewicz, the domain is [0, pi]^2.
    # The function has multiple local minima.
    # A common heuristic for LCB in BO is beta = 2 * log(N * pi^2 / (6*delta))
    # But with only 30 evals, we want strong exploration early.
    
    # Let's use a simple heuristic that decays exploration over the expected 25 calls.
    # We can track the call count using a static variable, which is safe for this
    # isolated function in a single-threaded evaluator.
    
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
        
    # Increment call count
    acquisition._call_count += 1
    t = acquisition._call_count
    
    # Total expected calls is 25.
    # We want beta to be large initially (high exploration) and decrease.
    # Let's map t from 1..25 to beta from ~4.0 down to ~1.0.
    # beta = 4.0 * (1 - (t-1)/24) + 1.0
    # At t=1: beta = 4.0
    # At t=25: beta = 1.0
    
    # However, standard LCB is: score = -mu + beta * sigma
    # High score means: low mu (good mean) or high sigma (high uncertainty).
    
    # Let's refine the beta schedule.
    # Early on, we need to explore widely. Michalewicz 2D is multimodal.
    # A beta of 2-4 is often good for early exploration.
    # As we approach the end, we should exploit more.
    
    # Linear decay:
    total_steps = 25
    if t > total_steps:
        t = total_steps
        
    # beta starts at 3.5 and ends at 0.5
    beta = 3.5 - (3.0 * (t - 1) / (total_steps - 1))
    
    # Calculate LCB score
    # We want to minimize f.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We return higher scores for better candidates.
    # So we return -LCB(x) = -mu(x) + beta * sigma(x)
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

import numpy as np
import time

# Global state for time management
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Strategy: GP-Lower Confidence Bound (GP-LCB).
    score = -mu - kappa * sigma
    
    Parameters:
    mu: np.ndarray of shape (2000,), posterior mean in objective units.
    sigma: np.ndarray of shape (2000,), posterior standard deviation in objective units.
    f_best: float, the lowest observed value so far.
    
    Returns:
    scores: np.ndarray of shape (2000,), higher scores indicate better candidates to evaluate next.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Check time budget. The evaluator kills at 330s. We add a small margin.
    # If we are close to the limit, we should return a safe result quickly.
    # However, GP-LCB is extremely fast (vectorized operations), so this is unlikely to be a bottleneck
    # unless the overhead of the Python call itself is high, which it isn't for 2000 floats.
    # Still, it's good practice to have a guard.
    if time.time() - _start_time > 320:
        # Fallback: just return -mu to ensure we don't crash or hang, though this shouldn't happen
        return -mu

    # GP-LCB for minimization:
    # We want to minimize f(x).
    # The lower confidence bound is LCB(x) = mu(x) - kappa * sigma(x).
    # We want to find the x that minimizes LCB(x).
    # Since the acquisition function must return scores where HIGHER is better,
    # we return -LCB(x) = -mu(x) + kappa * sigma(x).
    
    # kappa = 2.0 is a standard choice for exploration/exploitation balance.
    # A value of 2.0 corresponds to a 95% confidence interval in a 1D normal distribution (approx).
    kappa = 2.0
    
    # Calculate scores
    scores = -mu + kappa * sigma
    
    # Ensure finite values
    # If sigma is NaN or Inf, the score becomes invalid.
    # Replace non-finite scores with a very low value so they are not selected.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

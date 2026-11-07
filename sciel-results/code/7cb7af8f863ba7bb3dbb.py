import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Pure Lower Confidence Bound (LCB) acquisition function for Bayesian optimization.
    
    Uses a high exploration constant to aggressively target regions with high
    posterior variance, which is effective for the Michalewicz function where
    the global minimum is isolated and surrounded by high-gradient ridges.
    
    LCB = mu - k * sigma (for minimization; we want to minimize this, so
    higher acquisition score means lower LCB value, i.e., more promising).
    """
    start_time = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Check time before computation (though LCB is very fast)
    if time.time() - start_time > DEADLINE:
        return np.zeros_like(mu)
    
    # Pure LCB strategy with high exploration constant
    # For minimization: we want to find points where mu - k*sigma is small
    # Acquisition score = -(mu - k*sigma) = -mu + k*sigma
    # Higher score = more promising to evaluate
    k = 3.0  # High exploration constant to aggressively target high-variance regions
    
    score = -mu + k * sigma_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score

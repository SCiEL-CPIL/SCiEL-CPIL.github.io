"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking to respect the 330s wall-clock limit
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback: just pick the one with lowest mu / cost
        # to avoid crashing, though the run is likely over.
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy: Cost-aware Expected Improvement (EI).
    # For minimization, Improvement I = f_best - f(x).
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma, Phi is CDF, phi is PDF.
    
    # Calculate z
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Handle case where sigma is very small to avoid numerical issues, 
    # though sigma_safe handles division. 
    # If sigma is 0, EI is max(f_best - mu, 0).
    # The formula works if we are careful with limits.
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Cost adjustment: divide by cost to get "improvement per unit cost"
    score = ei / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END

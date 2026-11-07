"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Store start time to enforce deadline
_START_TIME = time.time()
# Deadline is 330s, keep a 5s margin
_DEADLINE = 305.0

def acquisition(mu, sigma, f_best):
    """
    Compute an acquisition score for each candidate point using UCB.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Check time limit
    if time.time() > _DEADLINE:
        # Return a neutral score if we are over time, though this should rarely happen
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Determine the number of evaluations performed so far.
    # We don't have direct access to the iteration count in the signature, 
    # but we can infer it from the magnitude of sigma or just use a fixed 
    # exploration constant. 
    # However, a common heuristic is to use a constant beta.
    # Let's use a fixed beta that is slightly more exploratory than typical defaults.
    # For minimization, UCB = -mu + beta * sigma.
    # Higher score means more promising (lower objective).
    # So we want to maximize -mu + beta * sigma.
    
    # A beta of 1.0 is standard. 
    # To encourage more exploration in the early stages (which is crucial for Hartmann 3D),
    # we can use a slightly higher beta. 
    # Since we don't know the iteration, we use a fixed value that balances exploration well.
    # 2.0 is a reasonable choice for a function with multiple minima and limited budget.
    beta = 2.0
    
    # Compute UCB score
    # We want to minimize f, so we maximize the "upper confidence bound" of the negative function
    # or equivalently, we look for points where the lower bound of the function is high?
    # No, UCB for minimization:
    # We predict f(x) ~ N(mu, sigma^2).
    # We want to find x that minimizes f(x).
    # The "Upper Confidence Bound" in the context of minimization is often defined as:
    # UCB(x) = mu(x) - beta * sigma(x)  (This is the Lower Confidence Bound for the function value,
    # but it represents the potential for low values if we are optimistic about the uncertainty).
    # Wait, let's be precise.
    # Standard UCB for maximization: UCB(x) = mu(x) + beta * sigma(x).
    # For minimization, we can maximize -f(x). Let g(x) = -f(x).
    # Then UCB_g(x) = mu_g(x) + beta * sigma_g(x) = -mu_f(x) + beta * sigma_f(x).
    # So we maximize -mu + beta * sigma.
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END

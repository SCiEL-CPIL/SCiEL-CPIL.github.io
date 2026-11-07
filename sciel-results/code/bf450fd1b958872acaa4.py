"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
# Set deadline to 320s to be safe against the 330s kill
_DEADLINE = _START_TIME + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Probability of Improvement (PI).
    
    Parameters:
    -----------
    mu : numpy.ndarray
        GP mean predictions, shape (2000,).
    sigma : numpy.ndarray
        GP standard deviations, shape (2000,).
    f_best : float
        Best observed function value so far.
    cost : numpy.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : numpy.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return zeros to indicate no preference, or random? 
        # Zeros are finite and valid.
        return np.zeros_like(mu)

    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI) with a small exploration bonus
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to encourage exploring uncertain regions
    # This helps escape local optima in the Rosenbrock valley
    exploration_bonus = 0.1 * sigma_safe
    
    # Cost-aware adjustment
    # We divide by cost to penalize expensive evaluations.
    # Cost is in [1, 2], so this is a mild penalty.
    cost_safe = np.maximum(cost, eps)
    
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

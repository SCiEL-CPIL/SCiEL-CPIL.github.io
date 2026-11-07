"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin

def _time_remaining():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Blended Probability of Improvement (BPI) for minimization.
    Combines the Probability of Improvement (PI) with a small exploration bonus 
    based on the probability density function (PDF) to balance exploitation and exploration.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior std at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Check deadline
    if _time_remaining() < 0.5:
        # Fallback: fast PI calculation
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        return pi
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Exploration bonus: PDF of the standardized improvement
    # This encourages evaluation in areas with high uncertainty (peaks of the PDF)
    phi = norm.pdf(z)
    
    # Blend: Primary focus on probability of improvement, slight exploration boost
    # The coefficient 0.1 is a heuristic to prevent excessive exploitation
    blended_score = pi + 0.1 * phi
    
    # Ensure finite output
    blended_score = np.nan_to_num(blended_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return blended_score
# EVOLVE-BLOCK-END

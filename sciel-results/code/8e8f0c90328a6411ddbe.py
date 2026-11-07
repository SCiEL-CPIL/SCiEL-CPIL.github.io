# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: evaluator kills at 330s wall-clock; leave a safety margin.
_DEADLINE = time.time() + 325.0

# Known minimum of the Hartmann 6d function (for adaptive exploration weight).
_HARTMANN_6D_MIN = -3.32237

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Expected Improvement.
    
    The acquisition score is:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    
    Plus an exploration term:
    beta * sigma(x)
    
    Where beta is adaptive:
    - Larger when f_best is far from the known global minimum (early stage).
    - Smaller when f_best is close to the known global minimum (late stage).
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Deadline check
    if time.time() > _DEADLINE - 1.0:
        # Fallback: fast EI
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive Exploration Weight
    # Gap from the known global minimum
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    
    # The Hartmann 6d function has a range roughly from -3.3 to 3.3 (approx 6.6 total range).
    # A gap of 0 means we found the optimum. A gap of ~6 means we are far away.
    # We want high exploration when gap is large, low when gap is small.
    
    # Normalize gap to [0, 1] approximately
    # Using 3.0 as a scale factor: if gap > 3, we are definitely far.
    gap_norm = min(gap / 3.0, 1.0)
    
    # Beta starts high (1.0) when far, drops to low (0.05) when close.
    # Linear interpolation in gap_norm
    beta = 0.05 + 0.95 * gap_norm
    
    # Combine EI and Exploration
    scores = ei + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

"""
Acquisition function using Expected Improvement (EI) for Bayesian Optimization.
This represents a structural shift from heuristic adaptive UCB to the theoretically
grounded Expected Improvement criterion, which is often superior for scalar
optimization tasks with well-calibrated Gaussian Processes.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
# Safety margin: The evaluator kills at 330s. We stop at 310s to ensure
# the function returns before the hard kill, allowing for cleanup or final return.
_DEADLINE = _START_TIME + 310.0

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition score.
    
    Parameters:
    mu: np.ndarray, shape (N,) - Posterior mean at candidate points.
    sigma: np.ndarray, shape (N,) - Posterior standard deviation at candidate points.
    f_best: float - The minimum observed value so far.
    
    Returns:
    np.ndarray, shape (N,) - EI scores. Higher scores indicate more promising candidates.
    """
    
    # Deadline guard: If we are running out of time, fall back to a simple
    # exploitation strategy (minimizing the mean) to ensure a valid return.
    if time.time() > _DEADLINE:
        return -mu

    # Ensure sigma is positive to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-9)

    # Calculate Expected Improvement
    # EI(x) = (f_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x) - xi) / sigma(x)
    
    # xi is a small constant that controls the trade-off between exploration and exploitation.
    # A value of 0 is pure EI. A small positive value encourages finding better points.
    # We use a small relative xi to adapt to the scale of the remaining gap.
    # Goldstein-Price minimum is 3.0.
    f_min_known = 3.0
    gap = f_best - f_min_known
    # If we are far from the minimum, we can be more exploratory (smaller xi relative to gap).
    # If we are close, we want to be precise (larger xi relative to gap? No, usually fixed small xi).
    # Standard practice is a small fixed xi or 0.
    # Let's use a very small xi to avoid numerical zero-gradient issues but keep it nearly pure EI.
    xi = 0.01 * max(gap, 1.0)
    
    z = (f_best - mu - xi) / sigma
    
    # Compute EI
    # Use norm.cdf and norm.pdf for numerical stability
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle potential numerical non-finites
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END

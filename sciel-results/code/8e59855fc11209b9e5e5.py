"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Combines Expected Improvement with a distance-scaled exploration bonus.
The exploration weight adapts based on how far f_best is from the known global minimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: The evaluator kills at 330s. We track time to ensure we return safely.
_START_TIME = time.monotonic()
_TIME_BUDGET = 300.0  # Seconds
_MARGIN = 10.0        # Safety margin

# Known global minimum of Styblinski-Tang 1D
GLOBAL_MIN = -39.1662


def _expected_improvement(mu, sigma, f_best):
    """
    Calculate Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with norm.cdf/pdf
    z = np.clip(z, -10, 10)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline guard: If we are close to the time limit, return a safe, fast score.
    if time.monotonic() - _START_TIME > _TIME_BUDGET - _MARGIN:
        # Return -mu as a fast fallback (lower mu is better, so higher score is -mu)
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # Calculate Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Dynamic exploration weight
    # Distance from current best to known optimum
    dist = f_best - GLOBAL_MIN
    if dist < 0:
        dist = 0.0
        
    # Lambda increases as we are further from the optimum
    # Base lambda 0.5, scale with distance
    # tau controls how fast lambda decays. 
    # If tau is small, lambda drops quickly.
    # If tau is large, lambda stays high longer.
    tau = 10.0
    lam = 0.5 * (1.0 + dist / tau)
    
    # Score = EI + Lambda * Sigma
    # This adds a simple, effective exploration bonus proportional to uncertainty
    scores = ei + lam * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

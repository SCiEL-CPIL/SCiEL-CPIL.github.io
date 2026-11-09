"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Gaussian Expected Improvement with
    exploration-exploitation balance and deadline guard.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    # Deadline guard: if we're close to the limit, return simple UCB to be fast
    if time.time() > DEADLINE - 5.0:
        # Fast fallback: UCB with moderate exploration
        sigma_safe = np.maximum(sigma, 1e-12)
        ucb = -mu + 1.0 * sigma_safe
        return np.nan_to_num(ucb, nan=0.0, posinf=1e6, neginf=-1e6)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Gaussian Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # CDF and PDF of standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add a small exploration term to balance exploitation and exploration
    # This helps escape local optima while still favoring improvement
    exploration = 0.3 * sigma_safe
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END

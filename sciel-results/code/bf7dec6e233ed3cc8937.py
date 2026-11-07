"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Strategy:
    - EI scores candidates by the expected amount of improvement over f_best.
    - For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
      where z = (f_best - mu) / sigma
    - This considers both probability AND magnitude of improvement.
    - A small uncertainty bonus helps exploration in early stages.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For minimization, improvement is f_best - f(x), so we want E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI is non-negative by construction, but clip for safety
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps escape local optima, especially in early iterations
    # Scale the bonus so it's meaningful but doesn't dominate
    exploration_bonus = 0.1 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

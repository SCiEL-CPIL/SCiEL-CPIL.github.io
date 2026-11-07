"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr, ndtr_inv
import time

# Deadline guard
_DEADLINE = 320.0
_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Probability of Improvement (PI) with exploration jitter.
    
    Uses the standard PI acquisition function which prioritizes points with a
    high probability of beating the current best. A small exploration jitter
    is added to encourage some exploration while primarily exploiting regions
    where improvement is likely. This avoids the heavy-tailed exploration of
    EI that can waste evaluations in high-dimensional spaces.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return finite scores that favor exploration
        return np.maximum(sigma, 1e-9)
    
    # Ensure sigma is safe (avoid division by zero)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the z-score: (f_best - mu) / sigma
    # Higher z means mu is well below f_best, so P(improvement) is high
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # where Phi is the standard normal CDF
    pi_score = ndtr(z)
    
    # Add a small exploration jitter proportional to sigma
    # This encourages visiting uncertain regions while still prioritizing
    # likely improvements
    epsilon = 1e-4
    exploration = epsilon * sigma_safe
    
    scores = pi_score + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state for time tracking
_start_time = None
_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def _get_elapsed():
    """Return elapsed time since module load, initializing on first call."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_deadline_exceeded():
    """Check if we are close to the 330s wall-clock limit."""
    return _get_elapsed() > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI is a robust acquisition function that balances exploration and exploitation.
    For minimization, we define the improvement as $f_{best} - f(x)$.
    
    Score to maximize: EI(x) = (f_{best} - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_{best} - mu(x)) / sigma(x)
    
    We include a small exploration constant xi to prevent premature convergence.
    """
    # Ensure sigma is positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Deadline guard: if we're close to the limit, use a fast fallback
    # We reuse the same logic but ensure it's fast.
    if _is_deadline_exceeded():
        # Fast path: use a simple heuristic if time is running out
        # Usually EI is fast enough, but just in case.
        z = (f_best - mu) / sigma_safe
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
        if not np.any(np.isfinite(ei_scores)):
            return np.zeros_like(mu)
        return ei_scores

    # Small exploration constant
    # As f_best gets closer to 0, the potential improvement shrinks.
    # A fixed xi might be too large or too small.
    # Let's use a small fixed xi to encourage finding slightly better points.
    xi = 0.01
    
    # Calculate z-score
    # z = (f_best + xi - mu) / sigma
    # If we use xi, the "target" is f_best + xi.
    # This encourages points that are at least xi better than current best.
    # If f_best is 0.1, target is 0.11. We look for mu < 0.11.
    # If f_best is 0.01, target is 0.02.
    
    z = (f_best + xi - mu) / sigma_safe
    
    # Handle numerical stability for large z
    # norm.cdf and norm.pdf are stable, but let's be safe.
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI formula
    # EI = (f_best + xi - mu) * Phi(z) + sigma * phi(z)
    ei_scores = (f_best + xi - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all scores are finite
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
    
    # If all are -inf, return zeros
    if not np.any(np.isfinite(ei_scores)):
        return np.zeros_like(mu)
    
    return ei_scores
# EVOLVE-BLOCK-END

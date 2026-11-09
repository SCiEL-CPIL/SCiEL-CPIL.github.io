"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 10.0  # 10 seconds margin

def _sanitize(mu, sigma):
    """Ensure inputs are finite and sigma is positive."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    return mu, sigma

def acquisition(mu, sigma, f_best):
    """
    Return 2000 finite scores; higher => evaluate next.
    
    Uses a Thompson Sampling proxy heuristic:
    score = (f_best - mu) / (sigma + epsilon)
    This encourages points with high potential improvement relative to uncertainty.
    """
    # Check time
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        # Return a deterministic fallback (e.g., based on mean only) to ensure validity
        mu, sigma = _sanitize(mu, sigma)
        return -mu

    mu, sigma = _sanitize(mu, sigma)
    
    # Calculate the normalized improvement score
    # f_best is the lowest observed value (lower is better)
    # We want to maximize (f_best - mu) / sigma
    # If mu < f_best, numerator is positive.
    # If sigma is small, score is large.
    eps = 1e-6
    scores = (f_best - mu) / (sigma + eps)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: If all scores are negative or very small, we might be in a
    # difficult region. But the heuristic naturally handles this by preferring
    # lower mu.
    
    return scores
# EVOLVE-BLOCK-END

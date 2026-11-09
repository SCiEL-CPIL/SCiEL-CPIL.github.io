"""Thompson sampling acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time across calls to enforce the 330s wall-clock budget
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # safety margin of 10s before the 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: sample one realization from the GP posterior
    at all 2000 candidates and return negative sampled values as scores.
    Higher score = more likely to be the minimum.
    """
    # Check deadline; if we're close to the limit, fall back to a simple
    # exploitation-based score (negative mu) to ensure we return valid output quickly.
    if time.time() > _DEADLINE - 0.5:
        # Exploitation fallback: just use -mu (higher means lower predicted value)
        return -mu

    # Ensure sigma is positive (avoid division issues)
    sigma_safe = np.maximum(sigma, 1e-12)

    # Sample a single realization from N(mu, sigma^2) for all candidates
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))

    # Return negative samples as scores: higher score = lower sampled value = more promising
    scores = -samples

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END

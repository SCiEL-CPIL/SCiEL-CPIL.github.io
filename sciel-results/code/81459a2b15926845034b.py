"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: keep a global start time to ensure we finish before the 330s limit.
_START_TIME = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill
_CALL_COUNT = 0


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Hartmann 3D BO.
    
    Instead of a deterministic acquisition (EI + heuristic exploration), we
    draw a single sample from the GP posterior at each candidate point and
    score candidates by how promising that sample is. This naturally balances
    exploration and exploitation: high-sigma regions get explored because
    their samples can be very good, while low-sigma regions are only chosen
    if their mean is already excellent.
    
    For minimization, we score by the negative of the sampled value, so
    higher score = better (lower) predicted function value.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Emergency: return EI to be fast and valid
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) at each candidate
    # Use a fixed seed based on call count for reproducibility within a run,
    # but different seeds across calls to get diverse samples.
    # Actually, we want randomness per call to get different exploration.
    # Use the default RNG for natural randomness.
    rng = np.random.default_rng(_CALL_COUNT * 1000 + 42)
    sample = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # For minimization, we want to pick the candidate with the lowest sampled value.
    # The acquisition returns scores where HIGHER means "evaluate next".
    # So we negate the sample: lower sampled f -> higher score.
    score = -sample
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

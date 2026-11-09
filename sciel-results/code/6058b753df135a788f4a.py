"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time
_ACQ_START = time.time()
_ACQ_DEADLINE = _ACQ_START + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with adaptive uncertainty penalty.
    
    Structural change: Instead of a fixed exploration weight, we use an
    adaptive penalty that scales with the signal-to-noise ratio. When
    uncertainty (sigma) is high relative to the expected improvement
    (f_best - mu), the exploration term is heavily penalized. When
    uncertainty is low and improvement is likely, the exploration term
    contributes more. This forces the search to prioritize high-confidence
    improvements in the sparse-data regime while still allowing some
    exploration in promising regions.
    
    The penalty factor is: exp(-alpha * sigma / max(f_best - mu, eps))
    which decays exponentially as the noise-to-signal ratio increases.
    """
    # Time guard: if we're close to the deadline, return a simple fallback
    if time.time() > _ACQ_DEADLINE:
        # Fallback: simple LCB-like score
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        cost = np.asarray(cost)
        fallback = (f_best - mu) / (cost + 1e-9)
        return np.nan_to_num(fallback, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon to avoid division by zero
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Calculate Z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Exploitation component: Expected gain if mu < f_best
    exploit_term = (f_best - mu) * cdf
    
    # Exploration component: Value of information
    explore_term = sigma * pdf
    
    # Adaptive penalty: penalize exploration when uncertainty is high
    # relative to the expected improvement
    # signal = f_best - mu (how much improvement is expected)
    # noise = sigma (uncertainty)
    # When noise >> signal, penalty is strong (exp(-large) ~ 0)
    # When noise << signal, penalty is weak (exp(-small) ~ 1)
    signal = np.maximum(f_best - mu, eps)
    noise_to_signal = sigma_safe / signal
    alpha = 1.5  # Controls how aggressively we penalize high-noise regions
    
    # Adaptive weight for exploration term
    adaptive_weight = np.exp(-alpha * noise_to_signal)
    
    # Combined Penalized EI with adaptive exploration weight
    ei_penalized = exploit_term + adaptive_weight * explore_term
    
    # Cost-aware score: Higher is better
    # Divide by cost to account for evaluation expense
    scores = ei_penalized / (cost + eps)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END

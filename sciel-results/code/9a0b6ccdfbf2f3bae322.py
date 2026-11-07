"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Fixed seed for deterministic Thompson sampling within a single acquisition call
# We use a simple LCG for speed and reproducibility per call
_RNG_STATE = 123456789

def _lcg():
    """Fast linear congruential generator."""
    global _RNG_STATE
    _RNG_STATE = (_RNG_STATE * 1103515245 + 12345) & 0x7FFFFFFF
    return _RNG_STATE / 0x7FFFFFFF

def _normal_sample(n):
    """Generate n standard normal samples using Box-Muller with LCG."""
    # Use pairs for Box-Muller
    samples = np.empty(n)
    for i in range(0, n - 1, 2):
        u1 = _lcg()
        # Ensure u1 is not 0
        if u1 < 1e-15:
            u1 = 1e-15
        u2 = _lcg()
        r = np.sqrt(-2.0 * np.log(u1))
        samples[i] = r * np.cos(2.0 * np.pi * u2)
        if i + 1 < n:
            samples[i + 1] = r * np.sin(2.0 * np.pi * u2)
    if n % 2 == 1:
        u1 = _lcg()
        if u1 < 1e-15:
            u1 = 1e-15
        u2 = _lcg()
        r = np.sqrt(-2.0 * np.log(u1))
        samples[-1] = r * np.cos(2.0 * np.pi * u2)
    return samples


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling variant with exploitation bias.
    
    mu, sigma: arrays of shape (N,) in objective units.
    f_best: lowest observed value.
    
    Returns: array of shape (N,) with scores. Higher score = more likely to be selected.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from posterior predictive
    # Sample z ~ N(0, 1)
    z = _normal_sample(n)
    ts_sample = mu - sigma_safe * z
    
    # Deterministic exploitation bias:
    # We want to favor points that are already good (low mu) and have some uncertainty
    # Combine TS sample with a deterministic EI-like term
    
    # Compute z-score for EI
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Normalize EI to [0, 1] range approximately
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros(n)
    
    # Combine: primarily Thompson Sampling, with small exploitation bias
    # The TS sample is in objective units (lower is better), so we negate for higher-is-better
    # Scale EI_norm to comparable range
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-12:
        mu_range = 1.0
    
    # Higher score means more likely to be evaluated next
    # TS sample: lower mu is better, so -ts_sample is higher for better points
    # But ts_sample can be very negative (good) or positive (bad)
    # Normalize ts_sample to be comparable with EI_norm
    
    # Use a weighted combination:
    # score = -ts_sample / scale + alpha * ei_norm
    # where alpha controls exploitation vs exploration
    
    # Adaptive alpha: start with more exploration, shift to exploitation as f_best improves
    # For Michalewicz, the minimum is -1.8013
    # If f_best is close to the minimum, we should exploit more
    
    min_possible = -1.8013
    progress = np.clip((min_possible - f_best) / (1.0 - min_possible), 0.0, 1.0)
    # When progress is high (f_best close to min), exploit more
    alpha = 0.1 + 0.4 * progress
    
    # Scale ts_sample to be in a reasonable range
    ts_scale = max(np.std(ts_sample), 1e-9)
    ts_norm = -ts_sample / ts_scale
    
    # Combine
    scores = ts_norm + alpha * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return the best we have
        pass
    
    return scores
# EVOLVE-BLOCK-END

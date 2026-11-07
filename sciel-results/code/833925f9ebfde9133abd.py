import numpy as np
import time
from scipy.special import erf, exp

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimizing the objective.
    
    This function computes the Expected Improvement for each candidate point.
    EI is a probabilistic ranking that balances exploration and exploitation
    by maximizing the expected gain over the current best observed value.
    
    mu, sigma : array-like, shape (2000,)
    f_best    : float, lowest value observed so far
    
    Returns: array of 2000 finite scores; HIGHER means "evaluate next".
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        # If we run out of time, just pick the point with the lowest predicted mean.
        score = f_best - mu
        return np.clip(score, -1e12, 1e12)

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for improvement
    # z = (f_best - mu) / sigma
    # If sigma is very small, z can be very large.
    # We clip z to prevent overflow in exp and erf calculations, although erf is stable.
    # exp(-z^2/2) underflows to 0 for z > ~38, which is fine (phi becomes 0).
    # However, (f_best - mu) * Phi(z) might have issues if z is inf.
    # Clipping z to [-10, 10] is safe because Phi(10) ~ 1, Phi(-10) ~ 0.
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Compute Phi(z) and phi(z) using scipy.special for speed and stability
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = (1 / sqrt(2*pi)) * exp(-z^2 / 2)
    
    sqrt_2 = np.sqrt(2.0)
    phi_coeff = 1.0 / np.sqrt(2.0 * np.pi)
    
    Phi = 0.5 * (1.0 + erf(z_clipped / sqrt_2))
    phi = phi_coeff * exp(-0.5 * z_clipped * z_clipped)
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) is positive if mu < f_best (potential improvement)
    
    improvement_term = (f_best - mu) * Phi
    uncertainty_term = sigma_safe * phi
    
    EI = improvement_term + uncertainty_term
    
    # EI should be non-negative. Due to numerical errors, it might be slightly negative.
    # Clip to 0 to ensure valid scores.
    EI = np.maximum(EI, 0.0)
    
    # Keep the output strictly finite and bounded.
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.clip(EI, 0.0, 1e12)
    
    # Guarantee exactly `n` finite entries (defensive).
    if EI.shape[0] != n:
        EI = np.resize(EI, n)
    
    return EI

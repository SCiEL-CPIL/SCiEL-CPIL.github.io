import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimizing the objective.
    
    EI(x) = E[max(f_best - f(x), 0)]
    
    This is a probabilistic ranking that directly maximizes the expected
    gain over the current best, inherently balancing exploration and
    exploitation based on the posterior distribution's tail probability.
    
    mu, sigma : array-like, shape (2000,)  (objective units, sigma >= 0)
    f_best    : float, lowest value observed so far

    Returns: array of 2000 finite scores; HIGHER means "evaluate next".
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        score = -mu
        return np.clip(score, -1e12, 1e12)

    # Handle potential numerical issues in sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # For very small sigma, EI should be approximately max(improvement, 0)
    # For very large sigma, EI should be approximately sigma * 0.4 (exploration)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is always non-negative; higher EI means better candidate
    # Clip to prevent overflow issues in downstream comparisons
    ei = np.clip(ei, 0.0, 1e12)
    
    # Guarantee exactly `n` finite entries (defensive).
    if ei.shape[0] != n:
        ei = np.resize(ei, n)
        
    return ei

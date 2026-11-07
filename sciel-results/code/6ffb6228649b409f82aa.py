import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimizing the objective.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Higher EI means the candidate is more likely to yield a new minimum.
    This balances exploration and exploitation based on the posterior tail probability.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        score = f_best - mu
        return np.clip(score, -1e12, 1e12)

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Compute z-score
    z = (f_best - mu) / sigma_safe

    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) is the CDF of the standard normal
    # phi(z) is the PDF of the standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip to prevent overflow issues
    ei = np.clip(ei, -1e12, 1e12)

    # Guarantee exactly `n` finite entries (defensive).
    if ei.shape[0] != n:
        ei = np.resize(ei, n)
        
    return ei

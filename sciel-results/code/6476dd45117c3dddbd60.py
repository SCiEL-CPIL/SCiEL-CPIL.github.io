import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimizing the objective.
    
    EI(x) = E[max(f_best - f(x), 0)]
         = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    where Phi and phi are the standard normal CDF and PDF.
    
    This uses the full posterior distribution to compute the expected gain,
    naturally balancing exploration (high sigma) and exploitation (low mu).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        score = f_best - mu
        score = np.where(np.isfinite(score), score, 0.0)
        return np.clip(score, -1e12, 1e12)

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute the normalized improvement: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi = standard normal CDF
    # phi = standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI is always non-negative; higher EI means better candidate
    score = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Keep the output strictly finite and bounded.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.clip(score, -1e12, 1e12)

    # Guarantee exactly `n` finite entries (defensive).
    if score.shape[0] != n:
        score = np.resize(score, n)

    return score

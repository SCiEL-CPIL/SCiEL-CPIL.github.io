"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).

    EI is a probabilistic acquisition that balances exploitation and exploration
    by estimating the expected improvement over the best observed value.
    It is structurally different from linear LCB because it uses the full
    Gaussian posterior tail rather than a fixed linear combination.

    Score = EI (higher is better)

    Args:
        mu: Array of GP mean predictions (2000,).
        sigma: Array of GP standard deviation predictions (2000,).
        f_best: The lowest observed function value.

    Returns:
        Array of EI scores (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()

    n = mu.shape[0]
    if n == 0:
        return np.array([])

    if time.time() > deadline:
        return np.zeros(n)

    # Sanitize sigma: non-negative and finite
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_clean = np.maximum(sigma_clean, 0.0)

    # Sanitize mu: replace non-finite with a large penalty value
    mu_clean = np.where(np.isfinite(mu), mu, 1e10)

    # Ensure f_best is finite; if not, fall back to mean of mu
    if not np.isfinite(f_best):
        finite_mu = mu_clean[np.isfinite(mu_clean)]
        if finite_mu.size > 0:
            f_best_val = float(np.min(finite_mu))
        else:
            f_best_val = 0.0
    else:
        f_best_val = float(f_best)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma_clean, eps)
    z = (f_best_val - mu_clean) / sigma_safe

    # Handle numerical stability: clip z to avoid extreme values
    z = np.clip(z, -10.0, 10.0)

    # Standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)

    # EI formula for minimization
    ei = (f_best_val - mu_clean) * Phi_z + sigma_clean * phi_z

    # For points with sigma near zero, EI should be max(f_best - mu, 0)
    # This is already handled approximately by the formula, but let's ensure non-negativity
    ei = np.maximum(ei, 0.0)

    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)

    return scores
# EVOLVE-BLOCK-END

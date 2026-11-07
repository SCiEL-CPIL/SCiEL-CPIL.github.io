import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    """
    Hybrid Thompson Sampling + Expected Improvement acquisition.
    
    Combines the stochastic exploration of Thompson Sampling (which enables
    jumps between multimodal basins in Michalewicz 2D) with a weighted
    Expected Improvement term that provides deterministic convergence pressure.
    
    The TS component samples from the posterior predictive f(x) ~ N(mu, sigma^2)
    and scores as -f_sample, naturally balancing exploration/exploitation.
    The EI component provides a smooth, deterministic pull toward regions where
    the posterior mean is already below f_best.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    np.ndarray, shape (N,)
        Acquisition scores; higher scores are preferred.
    """
    # Deadline guard: return safe scores if we're running out of time
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # --- Thompson Sampling component ---
    # Sample from posterior predictive: f(x) ~ N(mu(x), sigma(x)^2)
    # For minimization, score = -f_sample (higher is better)
    z = np.random.standard_normal(mu.shape)
    f_sample = mu + sigma_safe * z
    ts_score = -f_sample

    # --- Expected Improvement component ---
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z), where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    # Clip Z to avoid numerical overflow in norm.pdf/cdf
    Z = np.clip(Z, -50.0, 50.0)
    
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # EI is non-negative; it's positive when mu < f_best (improvement possible)
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    EI = np.maximum(EI, 0.0)  # Ensure non-negative

    # --- Combine: weighted blend ---
    # TS provides stochastic exploration to escape local minima
    # EI provides deterministic convergence toward known best
    # Weighting: TS dominates for exploration, EI adds convergence pressure
    w_ts = 0.75
    w_ei = 0.25

    # Normalize EI relative to TS score scale for balanced blending
    # Use the median absolute deviation of ts_score as a scale reference
    ts_scale = np.median(np.abs(ts_score))
    if ts_scale < 1e-15:
        ts_scale = 1.0
    ei_scale = np.median(np.abs(EI))
    if ei_scale < 1e-15:
        ei_scale = 1.0

    # Scale EI to be comparable in magnitude to TS
    ei_normalized = EI * (ts_scale / ei_scale)

    score = w_ts * ts_score + w_ei * ei_normalized

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score

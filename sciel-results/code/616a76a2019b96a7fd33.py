import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid Thompson Sampling / Expected Improvement
    approach for Bayesian optimization of the 2D Michalewicz function.
    
    Combines stochastic posterior sampling (for exploration and escaping local
    minima) with a deterministic Expected Improvement component (for exploitation
    and convergence pressure).
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior at candidate points
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed objective value so far
    
    Returns
    -------
    np.ndarray, shape (N,)
        Acquisition scores; higher scores are preferred.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample from posterior predictive distribution
    # For minimization, we want low values. Score = -(mu - sigma * z) = -mu + sigma * z
    # where z ~ N(0,1). Higher score means more likely to be a good (low) value.
    z = np.random.standard_normal(mu.shape)
    ts_score = -mu + sigma_safe * z
    
    # Compute Expected Improvement as a deterministic exploitation component
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Normalize both scores to [0, 1] range for stable combination
    # This prevents scale differences from dominating the blend
    
    ts_min, ts_max = ts_score.min(), ts_score.max()
    ei_min, ei_max = ei.min(), ei.max()
    
    ts_range = ts_max - ts_min if ts_max > ts_min else 1.0
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    
    ts_norm = (ts_score - ts_min) / ts_range
    ei_norm = (ei - ei_min) / ei_range
    
    # Blend Thompson sampling with EI exploitation
    # Weight primarily on TS for exploration, with a smaller EI component
    # for convergence pressure toward the known best
    w_ts = 0.8
    w_ei = 0.2
    
    score = w_ts * ts_norm + w_ei * ei_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If we're running out of time, return immediately
    if time.time() > deadline:
        return score
    
    return score

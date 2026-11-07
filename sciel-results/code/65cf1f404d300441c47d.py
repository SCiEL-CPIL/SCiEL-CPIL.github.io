"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Monte Carlo Expected Improvement (MC-EI)
    with a pure exploitation term (-mu).
    
    Strategy: 
    1. Estimate Expected Improvement via Monte Carlo sampling from the
       posterior N(mu, sigma^2). This is robust and avoids numerical
       instability in closed-form EI.
    2. Combine with -mu (exploitation) to ensure regions with very low
       predicted means are prioritized even when uncertainty is low.
    3. This creates a natural exploration/exploitation balance:
       - Early on (high sigma), MC-EI dominates, encouraging exploration.
       - Late on (low sigma), -mu dominates, encouraging exploitation.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        # Return a safe fallback: just negative mu (exploitation only)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Monte Carlo Expected Improvement ---
    # We want to estimate E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    
    # Number of samples per point.
    # 100 samples is a good trade-off between accuracy and speed for 2000 points.
    # Total operations: 2000 * 100 = 200,000, which is very fast in NumPy.
    K = 100
    
    # Generate random samples: shape (K, n)
    # Standard normal samples
    z = np.random.standard_normal((K, n))
    
    # Transform to predicted values: f_sample = mu + sigma * z
    # mu and sigma are (n,), so we broadcast
    f_samples = mu[None, :] + sigma_safe[None, :] * z
    
    # Compute improvement: max(f_best - f_samples, 0)
    improvements = np.maximum(f_best - f_samples, 0.0)
    
    # Average over samples to get MC-EI estimate: shape (n,)
    mc_ei = np.mean(improvements, axis=0)
    
    # --- Exploitation Term ---
    # We want to reward low mu values.
    # -mu is high (positive) when mu is low (good).
    # -mu is low (negative) when mu is high (bad).
    
    # Combine: score = -mu + mc_ei
    # This naturally adapts:
    # - When sigma is large, mc_ei is large, exploration is encouraged.
    # - When sigma is small, mc_ei ~ 0, -mu dominates, exploitation is encouraged.
    
    scores = -mu + mc_ei
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END

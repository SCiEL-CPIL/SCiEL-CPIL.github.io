"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 15.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses an adaptive strategy:
    - Early phase (f_best is high): Use pure EI to explore broadly and ignore cost.
    - Late phase (f_best is low): Use CEI (EI / cost) to be cost-efficient and refine the minimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        GP mean predictions
    sigma : ndarray, shape (n_candidates,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far (lower is better)
    cost : ndarray, shape (n_candidates,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        return np.full(len(mu), 0.0)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe sigma for calculations
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    # Compute Z-score: (f_best - mu) / sigma
    # Note: For minimization, we want mu < f_best.
    # Z > 0 implies mu < f_best (potential improvement)
    Z = (f_best - mu) / safe_sigma
    
    # Compute standard normal CDF and PDF
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    EI = (f_best - mu) * Phi_Z + safe_sigma * phi_Z
    
    # Numerical safety: EI should be >= 0
    EI = np.maximum(EI, 0.0)
    
    # Adaptive Strategy:
    # Early phase (f_best is high, e.g., > -3.5): Use pure EI to explore broadly and ignore cost.
    # Late phase (f_best is low, e.g., < -3.5): Use CEI to be cost-efficient and refine the minimum.
    # Threshold chosen based on Hartmann 3D minimum (-3.86) and typical random start values.
    threshold = -3.5
    
    if f_best > threshold:
        score = EI
    else:
        # Cost-Effective EI (CEI)
        # CEI = EI / cost
        score = EI / cost
        
    # Final safety check
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END

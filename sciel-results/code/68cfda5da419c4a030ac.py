import numpy as np
import time
from scipy.special import erfc, log_ndtr, ndtr

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    Uses EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.
    The score is EI / cost, which prioritizes points with high expected improvement
    relative to their evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a safe fallback: prefer cheap candidates with low predicted mean
        mu_arr = np.asarray(mu, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.clip(cost_arr, 1e-10, None)
        fallback = np.maximum(f_best - mu_arr, 0.0) / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and ensure positivity
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid underflow/overflow in PDF/CDF
    # For very large positive z, Phi(z) -> 1, phi(z) -> 0
    # For very large negative z, Phi(z) -> 0, phi(z) -> 0
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Phi(z) and phi(z) using scipy.special for numerical stability
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    phi_z = 0.5 * erfc(-z_clipped / np.sqrt(2.0))
    
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    # Use log-space for stability: log(phi(z)) = -0.5*z^2 - 0.5*log(2*pi)
    log_phi_z = -0.5 * z_clipped * z_clipped - 0.5 * np.log(2.0 * np.pi)
    phi_z_pdf = np.exp(log_phi_z)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * phi_z + sigma_safe * phi_z_pdf
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

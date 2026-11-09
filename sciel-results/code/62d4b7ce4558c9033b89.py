"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing Griewank 2D using Expected Improvement (EI).
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best (lowest) observed value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # f_best is the best observed value (lowest since we minimize)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0
    
    # Strategy: Cost-aware Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This captures both the probability of improvement AND the magnitude of expected improvement,
    # providing a more informative signal than PI alone for optimization.
    
    # Compute Z-score
    z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    phi = norm.pdf(z)  # Standard normal PDF
    Phi = norm.cdf(z)  # Standard normal CDF
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # EI can be negative if mu > f_best significantly; clip to 0
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END

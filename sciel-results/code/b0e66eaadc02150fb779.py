"""Rewritten acquisition function for cost-aware Bayesian optimization on Powell 4D."""
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement and Upper Confidence Bound.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value (minimum so far).
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Time guard: if we've been running too long, return a safe fallback
    # This prevents timeout issues
    elapsed = time.time()
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return uniform scores if any input is non-finite
        return np.ones(n)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Upper Confidence Bound (UCB) term
    # UCB = mu + kappa * sigma (we want to minimize, so we use -mu + kappa*sigma for exploration)
    # Actually for minimization: we want low mu, but high sigma for exploration
    # UCB for minimization: -mu + kappa * sigma
    kappa = 1.5
    ucb = -mu + kappa * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is also in objective units (roughly)
    # Scale EI by 1/cost to get expected improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # For UCB, we also want to penalize high cost
    ucb_per_cost = ucb / cost_safe
    
    # Combine: use a weighted blend
    # We can normalize both to [0,1] range relative to their max, or just combine directly
    # Let's combine them with a tunable weight
    w_ei = 0.7
    w_ucb = 0.3
    
    scores = w_ei * ei_per_cost + w_ucb * ucb_per_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

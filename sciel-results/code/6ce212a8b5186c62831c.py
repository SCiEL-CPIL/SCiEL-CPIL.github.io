"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function for minimization.
    
    This strategy uses a simple, robust LCB approach:
    score = (mu - kappa * sigma) / cost
    
    where kappa is chosen adaptively based on the spread of sigma values.
    Using LCB instead of EI provides a different exploration/exploitation trade-off
    that may perform better on these specific test functions.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    if _check_deadline():
        # Fallback: simple deterministic LCB if we're running out of time
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        kappa = 2.0
        scores = (mu - kappa * sigma_safe) / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma and cost are strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive kappa: use a moderate exploration coefficient
    # We can scale kappa based on the ratio of f_best to the mean of mu
    # to adjust exploration intensity based on how close we are to convergence
    mu_mean = np.mean(mu)
    # If f_best is much lower than mu_mean, we're likely converged, reduce exploration
    # Otherwise, increase exploration
    spread = np.std(sigma_safe)
    if spread > 1e-6:
        # Normalize kappa based on the relative uncertainty
        kappa = 2.0 * (1.0 + 0.5 * np.exp(-spread / (np.abs(f_best - mu_mean) + 1e-9)))
    else:
        kappa = 1.0
    
    # LCB for minimization: we want to minimize mu - kappa * sigma
    # So the acquisition score (higher is better) is mu - kappa * sigma
    # Divide by cost to account for evaluation cost
    scores = (mu - kappa * sigma_safe) / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

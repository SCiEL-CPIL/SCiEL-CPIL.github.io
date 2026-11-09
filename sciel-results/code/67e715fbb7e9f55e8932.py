"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural combination of
    Expected Improvement (EI) and a lower-confidence-bound (LCB) style
    exploration term, both normalized by cost.
    
    The LCB term provides a different exploration signal than raw sigma:
    it penalizes high mu (bad predicted values) while rewarding high sigma
    (uncertainty), creating a more targeted exploration than pure sigma.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time and return early if approaching limit
    start_time = time.time()
    
    # Ensure arrays are float64 for numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma and cost to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # LCB-style exploration term:
    # The LCB acquisition for minimization is: mu - kappa * sigma
    # We want higher scores to be better, so we use: -mu + kappa * sigma
    # This rewards points with low predicted value AND high uncertainty.
    # The kappa parameter controls exploration-exploitation tradeoff.
    # A value around 1.0-2.0 is typical; we use 1.0 for balanced exploration.
    kappa = 1.0
    lcb = -mu + kappa * sigma_safe
    
    # Normalize LCB to be comparable in scale to EI
    # Use the range of LCB values for normalization
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-9:
        lcb_norm = (lcb - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb)
    
    # Cost-normalized LCB
    lcb_cost = lcb_norm / cost_safe
    
    # Combined score: weighted sum of EI and LCB, both cost-normalized
    # EI weight: 0.7 (exploitation-focused)
    # LCB weight: 0.3 (exploration-focused, but targeted)
    ei_weight = 0.7
    lcb_weight = 0.3
    
    scores = ei_weight * ei_cost + lcb_weight * lcb_cost
    
    # Check deadline: if we're close to the 300s limit, return what we have
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Return current scores as-is (they should already be computed)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

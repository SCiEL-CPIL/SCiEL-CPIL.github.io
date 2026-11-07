"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: total wall-clock budget with safety margin
_DEADLINE = 320.0  # seconds; evaluator kills at 330s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions in objective units
        sigma: array of shape (2000,), GP standard deviations in objective units
        f_best: scalar, lowest observed function value
        cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
        array of shape (2000,), finite scores (higher = evaluate next)
    """
    start_time = time.time()
    
    # Ensure inputs are 1-D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score: how many std devs below f_best each point is
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # Cost-aware combination:
    # Use a blend of EI and POI, normalized by cost.
    # POI is more robust for exploration in the Rosenbrock valley.
    # We use a weighted combination that emphasizes POI when uncertainty is high.
    
    # Normalize EI to [0, 1] range for blending
    ei_max = np.max(ei)
    if ei_max > 1e-15:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend: 0.5 * POI + 0.5 * EI_norm, then divide by cost
    # This gives a structural change from pure EI/cost to a hybrid that
    # favors exploration (POI) while still rewarding expected improvement.
    score = (0.5 * poi + 0.5 * ei_norm) / np.maximum(cost, 1e-9)
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: if we're running long, return immediately
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 10:
        return score
    
    return score
# EVOLVE-BLOCK-END

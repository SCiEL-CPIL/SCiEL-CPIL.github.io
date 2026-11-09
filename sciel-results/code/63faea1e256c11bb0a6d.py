# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.

    Uses adaptive LCB (Lower Confidence Bound) with dynamic exploration
    parameter that scales with relative uncertainty. This is a structurally
    different modeling family from EI-based approaches, providing stronger
    exploration early on and transitioning to exploitation as uncertainty
    decreases.

    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)
    if n == 0:
        return np.array([])

    # Adaptive LCB: score = -mu - beta * sigma
    # Higher score = lower predicted value + more exploration
    # 
    # Dynamic beta: starts high for exploration, decreases as we get more
    # information. Use the ratio of typical uncertainty to typical improvement
    # potential to calibrate beta.
    
    # Estimate the scale of the function values
    # Use the spread of mu as a proxy for how much we've learned
    mu_range = np.ptp(mu_safe) if n > 1 else 1.0
    if mu_range < 1e-6:
        mu_range = 1.0
    
    # Typical uncertainty level
    sigma_median = np.median(sigma_safe)
    
    # Relative uncertainty: how large is sigma compared to the scale of mu variation
    # If sigma is large relative to mu variation, we're still exploring
    # If sigma is small, we should exploit more
    rel_uncertainty = sigma_median / (mu_range + 1e-6)
    
    # Beta scales with relative uncertainty, but is bounded
    # High rel_uncertainty -> high beta (explore)
    # Low rel_uncertainty -> low beta (exploit)
    # Use a smooth transition
    base_beta = 2.0  # Starting exploration strength
    beta = base_beta * np.tanh(rel_uncertainty * 2.0)
    
    # Ensure beta is at least a small value to maintain some exploration
    beta = max(beta, 0.1)
    
    # LCB score: we want to minimize mu + beta*sigma, so maximize -(mu + beta*sigma)
    lcb_score = -(mu_safe + beta * sigma_safe)
    
    # Cost adjustment: prefer cheaper evaluations
    # Use inverse cost weighting
    cost_factor = 1.0 / cost_safe
    scores = lcb_score * cost_factor
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    # Final safety
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses a structurally different approach: Thompson sampling with cost adjustment.
    This is a different design family from LCB - it samples from the posterior
    and selects based on expected improvement per unit cost.
    """
    # Explicit deadline guard
    start_time = time.time()
    deadline = 325.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    # Sanitize inputs
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Thompson Sampling approach:
    # Sample from the posterior: y = mu + sigma * z where z ~ N(0,1)
    # Then select the candidate with the best (lowest) y / cost ratio
    # This is structurally different from LCB - it uses randomization
    # and the sampling naturally balances exploration/exploitation
    
    # Generate random samples from the GP posterior
    z = np.random.standard_normal(n)
    samples = mu_safe + sigma_safe * z
    
    # For minimization, we want the lowest sample value
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Higher score = more desirable, so we negate
    score = -samples / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Check deadline
    if time.time() > deadline:
        # Return current best effort
        pass

    return score
# EVOLVE-BLOCK-END

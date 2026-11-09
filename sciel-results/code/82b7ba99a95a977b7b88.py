"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    
    Uses additive form: Score = -mu + beta * sigma - lambda * (cost - cost_mean)
    This is structurally different from EI/cost ratios, providing more stable
    exploration-exploitation-cost trade-off.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: favor low cost and high uncertainty
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return -cost_safe + 0.1 * sigma_safe
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe versions for computation
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive exploration coefficient based on current uncertainty level
    # When max sigma is high, we're still exploring -> use higher beta
    # When max sigma is low, we're exploiting -> use lower beta
    max_sigma = np.max(sigma_safe)
    
    # Beta scales with uncertainty: high uncertainty -> more exploration
    # Base beta ~2.5 for 8D with ~20 evals, scaled by relative uncertainty
    # Normalize max_sigma: if it's near the expected initial scale (~1.0 for cosine8), beta is high
    # As we learn, sigma decreases, beta decreases
    beta = 2.0 + 1.5 * min(max_sigma / 1.0, 1.0)
    
    # Cost penalty: subtract lambda * (cost - cost_mean) to center the penalty
    # This avoids systematic bias toward low-cost or high-cost regions
    cost_mean = np.mean(cost_safe)
    lambda_cost = 0.3  # Cost penalty weight
    
    # UCB score: -mu + beta * sigma (for minimization, we maximize this)
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware: subtract centered cost penalty
    scores = ucb_score - lambda_cost * (cost_safe - cost_mean)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

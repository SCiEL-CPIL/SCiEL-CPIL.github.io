"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty bonus for exploration
    3. Cost-awareness via log-normal cost model
    
    The cost model treats cost as log-normally distributed around the deterministic cost,
    which better handles cost uncertainty in low-data regimes.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance from best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-awareness: use a log-normal cost model
    # The deterministic cost is given; we model actual cost as log-normal
    # with mean = cost and some variance to account for uncertainty
    # E[1/cost] for log-normal is approximately 1/(cost * exp(var/2))
    # We use a moderate variance to be conservative
    cost_var = 0.1  # variance of log(cost)
    cost_mean_log = np.log(np.maximum(cost, 1e-6))
    # E[1/cost] = exp(-mean_log + var/2)
    inv_cost_expected = np.exp(-cost_mean_log + cost_var / 2.0)
    
    # Base score: EI per unit expected cost
    base_score = ei * inv_cost_expected
    
    # Exploration bonus: reward high uncertainty regions
    # This helps escape local optima and explore the space
    # Scale uncertainty bonus relative to the problem scale
    # Shekel 4d has range ~ [-10.5, 10], so scale by ~20
    exploration_weight = 0.5
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combine: weighted sum of exploitation and exploration
    # The weight balances between finding good points and exploring
    # Use a soft combination that emphasizes exploitation when EI is high
    combined = base_score + exploration_bonus
    
    # Ensure finite values
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    # Clip to reasonable range to avoid numerical issues
    max_val = np.percentile(combined, 99.5) if len(combined) > 0 else 0
    if np.isfinite(max_val) and max_val > 0:
        combined = np.clip(combined, 0, max_val * 2)
    
    return combined
# EVOLVE-BLOCK-END

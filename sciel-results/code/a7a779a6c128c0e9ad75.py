# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s. We track elapsed time from module load.
_t0 = time.time()
_DEADLINE = 325.0  # safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions in objective units
        sigma: (2000,) array of GP std dev predictions in objective units
        f_best: scalar, lowest observed function value so far
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of acquisition scores (higher = better)
    """
    # Time check: if we're close to deadline, return a simple baseline quickly
    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Structural change: Use a hybrid PI/EI acquisition that adapts based on uncertainty.
    # When sigma is high, use EI for exploration. When sigma is low, use PI for exploitation.
    # This avoids EI's stalling when sigma is small but mu is close to f_best.
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI)
    xi = 0.01  # small exploration parameter
    z_ei = (f_best - mu - xi) / sigma_safe
    ei = (f_best - xi - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Compute Probability of Improvement (PI)
    z_pi = (f_best - mu) / sigma_safe
    pi = norm.cdf(z_pi)
    
    # Adaptive blending: use PI when sigma is small (exploitation), EI when sigma is large (exploration)
    # Normalize sigma to get a blending weight
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-10:
        # Blend weight: 1 when sigma is large (use EI), 0 when sigma is small (use PI)
        blend_weight = sigma_safe / sigma_max
        # Smooth the transition
        blend_weight = np.clip(blend_weight, 0.0, 1.0)
    else:
        blend_weight = np.zeros_like(sigma_safe)
    
    # Hybrid acquisition: weighted combination of PI and EI
    # PI gives a probability in [0,1], EI gives an expected improvement
    # Scale PI to be comparable to EI
    scale_pi = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    pi_scaled = pi * scale_pi
    
    # Combine: blend between PI (exploitation) and EI (exploration)
    hybrid = blend_weight * ei + (1.0 - blend_weight) * pi_scaled
    
    # Cost-aware penalty: prefer cheaper evaluations
    # Cost is in [1, 2] range, so normalize to [0, 1]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range  # [0, 1], lower is better
    else:
        cost_norm = np.zeros_like(cost)
    
    # Apply cost penalty: multiply by (1 - cost_norm)^gamma
    gamma = 1.0
    cost_factor = (1.0 - cost_norm) ** gamma
    
    # Final score
    score = hybrid * cost_factor
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

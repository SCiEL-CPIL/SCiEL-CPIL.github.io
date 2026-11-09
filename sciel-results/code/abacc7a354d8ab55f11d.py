"""Cost-aware acquisition for Bayesian optimization of shekel 4d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses a cost-normalized expected improvement with an exploration bonus.
    The cost structure is 1 + mean(x) where x is in [0,10]^4, so cost ranges
    from 1 to 6. We use a soft cost penalty that's more stable than division.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid extreme values that cause numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected improvement (standard formulation)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost penalty: use a power-based penalty that's more stable than division
    # cost is in [1, 6], so we normalize it
    cost_norm = cost / np.mean(cost)  # Normalize to mean 1
    
    # Use a soft penalty: divide by cost but add a small constant for stability
    # This avoids the instability of pure division when cost varies
    cost_penalty = cost_norm ** 0.5  # Square root makes it less aggressive
    
    # Base acquisition: EI per unit cost (softened)
    base_score = ei / (cost_penalty + 1e-6)
    
    # Add a small exploration bonus to encourage exploring uncertain regions
    # This helps when EI is small but uncertainty is high
    exploration_bonus = 0.05 * sigma_safe / cost_penalty
    
    # Combine: primary signal is cost-normalized EI, secondary is exploration
    score = base_score + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with exploration weighting.
    # Use a small jitter to avoid division by zero.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize cost to [0, 1] roughly (cost is 1 + mean(x_scaled) so in [1, 2])
    # Divide by cost to get EI per unit cost.
    # To improve exploration, we can weight by sigma or use a different metric.
    # However, the seed uses EI/cost.
    # Let's try a slightly different approach: 
    # The cost is 1 + mean(x/10). So cost is in [1, 2].
    # Lower cost is better.
    # We want to maximize (improvement) / cost.
    
    # One issue with simple EI/cost is that it might be too greedy.
    # Let's add a small exploration bonus based on sigma.
    # But the directive says "KEEP this design". The design is "Expected improvement per unit cost".
    
    # Perhaps the issue is numerical stability or the specific form of EI.
    # Let's ensure z is clipped to avoid extreme values in cdf/pdf if sigma is very small.
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost is in [1, 2].
    # If cost is lower, we get more points for the same EI.
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a structural information-gain model.
    
    Instead of EI/cost (ratio of expectations), we model the stochastic
    knapsack structure directly:
    - Exploitation: probability of improvement (PI) scaled by expected gain
    - Exploration: GP uncertainty in low-cost regions (where shifted minima
      are likely to reside in the inner 80% of the domain)
    - Cost suppression: exponential penalty for high-cost regions
    
    The cost structure 1 + mean(x_scaled_to_01) means:
    - Minimum cost = 1 (at x = [-10, -10] -> scaled [0,0])
    - Maximum cost = 2 (at x = [10, 10] -> scaled [1,1])
    - The shifted minimum is in the inner 80%, i.e., x in [-8, 8]^2,
      which corresponds to scaled [0.1, 0.9]^2, cost in [1.1, 1.9]
    """
    # Clamp sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Standardized distance from current best
    z = (f_best - mu) / sigma
    
    # Probability of improvement (0 to 1)
    pi = norm.cdf(z)
    
    # Expected improvement (non-negative)
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware suppression: exponential decay for expensive regions
    # cost ranges from ~1 to ~2; we want to suppress high-cost regions
    # Use a soft exponential: exp(-lambda * (cost - 1))
    # At cost=1: factor=1; at cost=2: factor=exp(-lambda)
    # Choose lambda so that cost=2 is suppressed to ~0.3
    lambda_cost = 1.2
    cost_factor = np.exp(-lambda_cost * (cost - 1.0))
    
    # Exploration bonus: GP uncertainty weighted by cost efficiency
    # Higher sigma in low-cost regions is more valuable
    # The inner 80% region (where minima are) has cost in [1.1, 1.9]
    # We reward uncertainty in this band
    # Use a Gaussian-like weight centered at cost ~1.5 (middle of inner region)
    cost_band_weight = np.exp(-2.0 * (cost - 1.5) ** 2)
    
    # Exploration term: sigma * cost_band_weight * cost_factor
    # This rewards high uncertainty in the likely-minimum region
    exploration = sigma * cost_band_weight * cost_factor
    
    # Exploitation term: EI * cost_factor
    # This rewards expected improvement, discounted by cost
    exploitation = ei * cost_factor
    
    # Combine: weighted sum with emphasis on exploitation when f_best is close to 0
    # As f_best approaches 0, we should focus more on exploitation
    # Use a smooth transition: when f_best is large, explore more
    f_best_abs = abs(f_best)
    # When f_best is large (early), explore more; when small (late), exploit more
    explore_weight = 0.3 + 0.7 * min(f_best_abs / 10.0, 1.0)
    exploit_weight = 1.0 - explore_weight
    
    score = exploit_weight * exploitation + explore_weight * exploration
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

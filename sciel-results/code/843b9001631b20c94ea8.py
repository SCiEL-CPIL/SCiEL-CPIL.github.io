"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for stateless acquisition: we can't track calls, so we use a simple heuristic.
# The key insight: Goldstein-Price has multiple local minima. We want to balance
# exploitation (near f_best) and exploration (in unexplored regions).

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement with an exploration bonus
    based on uncertainty, plus a small randomization to escape plateaus.
    
    mu: (N,) array of GP mean predictions
    sigma: (N,) array of GP standard deviation predictions
    f_best: lowest observed function value
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: reward high uncertainty regions
    # Scale the bonus relative to the typical EI values
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    # Uncertainty-based exploration: sigma relative to the range of sigma
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        sigma_normalized = sigma / sigma_max
    else:
        sigma_normalized = sigma
    
    # Combine: primarily EI, with a moderate exploration component
    # The exploration term helps escape local optima in multimodal landscapes
    exploration_weight = 0.15
    score = ei_normalized + exploration_weight * sigma_normalized * (1.0 - ei_normalized)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

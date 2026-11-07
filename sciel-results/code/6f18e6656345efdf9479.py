"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines expected improvement (EI) with a cost-awareness mechanism.
    The key insight is that we want to maximize improvement per unit cost,
    but also need to balance exploitation vs exploration.
    
    We use a modified acquisition that:
    1. Computes expected improvement (EI)
    2. Scales by 1/cost to prefer cheaper evaluations
    3. Adds an exploration bonus weighted by uncertainty
    4. Uses a smooth, bounded formulation for numerical stability
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to avoid extreme values
    ei = np.clip(ei, 0.0, None)
    
    # Cost-aware: improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # Exploration term: encourage visiting uncertain regions
    # Scale by a small factor to not dominate the exploitation term
    # Use a smooth function of sigma to keep things differentiable
    exploration = 0.005 * sigma_safe / (sigma_safe + 0.1)
    
    # Combine exploitation and exploration
    scores = ei_per_cost + exploration
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

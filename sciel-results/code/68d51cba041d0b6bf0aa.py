"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses a hybrid of Expected Improvement per unit cost and a stochastic
    probability-of-improvement weighted by cost efficiency, with a small
    exploration bonus for regions of high uncertainty that are cheap to evaluate.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative sigma/cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standard EI computation
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Cost-adjusted EI (primary signal)
    ei_per_cost = ei / cost
    
    # Probability of improvement (exploration signal)
    poi = cdf
    
    # Weighted combination: favor high EI/cost, but also consider PoI for cheap points
    # The idea: if a point is very cheap (low cost), even moderate EI is worth taking
    # We use a normalized cost efficiency factor
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cheapness bonus: points with lower cost get a slight boost
    cheapness = 1.0 - cost_norm
    
    # Combine: primary is EI/cost, secondary is PoI * cheapness
    # Scale PoI contribution to be comparable to EI/cost
    ei_scale = np.max(np.abs(ei_per_cost)) + 1e-12
    poi_contribution = poi * cheapness * 0.1 * ei_scale
    
    score = ei_per_cost + poi_contribution
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: combine Expected Improvement with a
    # deterministic penalty based on the current cost.  This is a structural
    # change from pure EI/cost: we treat cost as a hard constraint-like penalty
    # in a constrained-optimisation style (like constrained EI or EI with
    # penalty), rather than a simple division.
    
    # Clamp sigma and cost for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Penalise expensive evaluations: use a quadratic penalty scaled by how
    # much cost exceeds the minimum cost, so that cheap points are strongly
    # preferred when EI is similar.
    c_min = np.min(cost)
    c_range = np.max(cost) - c_min
    if c_range < 1e-12:
        c_norm = np.zeros_like(cost)
    else:
        c_norm = (cost - c_min) / c_range  # in [0, 1]
    
    # Scale EI to the same order as the penalty term
    ei_scale = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ei_scaled = ei / ei_scale
    
    # Penalty: quadratic in normalised cost, weighted so that the most
    # expensive points are heavily penalised.  The weight is chosen so that
    # at c_norm=1 the penalty is comparable to the max EI.
    penalty_weight = 1.5
    scores = ei_scaled - penalty_weight * c_norm**2
    
    # Also add a small exploration bonus for points with high uncertainty
    # relative to the improvement, to avoid over-exploitation of low-sigma
    # regions that may still be far from the optimum.
    z_safe = np.clip(z, -10, 10)
    unc_bonus = 0.05 * sigma / (np.abs(f_best) + 1e-6) * norm.pdf(z_safe)
    scores = scores + unc_bonus
    
    # Guarantee finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END

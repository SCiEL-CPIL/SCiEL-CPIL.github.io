"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with a small exploration floor and robust numerics.
    #
    # 1) Guard against zero/negative sigma.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 2) Compute standard EI: (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu)/sigma
    delta = f_best - mu
    z = delta / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = delta * Phi_z + sigma_safe * phi_z
    
    # 3) Cost-normalize: divide by cost (cost is in [1, 2]).
    #    This favors cheaper points (lower x_scaled, i.e., lower-left region) equally per unit cost.
    cost_safe = np.maximum(cost, eps)
    score_ei = ei / cost_safe
    
    # 4) Add a tiny exploration term proportional to sigma / cost to prevent premature convergence
    #    and encourage sampling in uncertain regions, especially early on.
    #    The coefficient is small so it doesn't dominate exploitation.
    exploration = 0.05 * sigma_safe / cost_safe
    
    score = score_ei + exploration
    
    # 5) Ensure all values are finite.
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END

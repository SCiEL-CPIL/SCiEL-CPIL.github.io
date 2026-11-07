"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware EI with cost-dependent exploration boost.
    
    The exploration weight uses a sigmoid of normalized cost so that
    cheaper regions (lower cost) receive relatively more exploration
    signal, while expensive regions are dominated by exploitation.
    """
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Base expected improvement (in objective units)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost normalization: cost ranges roughly [1, 2] (1 + mean of two coords in [0,1])
    # Map cost to [0, 1] for sigmoid: cheap -> 0, expensive -> 1
    cost_norm = np.clip((cost - 1.0) / 1.0, 0.0, 1.0)
    
    # Sigmoid exploration weight: cheap regions get higher exploration
    # beta controls the strength of the cost-dependent exploration modulation
    beta = 1.5
    expl_weight = 1.0 + beta * np.exp(-cost_norm * 3.0)
    
    # Cost-normalized acquisition with exploration boost
    # The exploration term is scaled by sigma (uncertainty) and boosted for cheap regions
    score = ei / cost + expl_weight * sigma / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END

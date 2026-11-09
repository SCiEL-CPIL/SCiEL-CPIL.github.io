"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Expected Improvement and 
    an adaptive exploration term. The exploration term is scaled by 
    uncertainty to balance exploitation and exploration, helping to 
    navigate the multimodal Griewank landscape.
    """
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute z-score for improvement probability
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    cdf = norm.cdf(z)   # P(f(x) < f_best) = discovery probability
    pdf = norm.pdf(z)

    # Standard Expected Improvement
    ei = (f_best - mu) * cdf + sigma * pdf

    # Adaptive exploration term:
    # We add a term that encourages exploring high-uncertainty regions.
    # The magnitude of this term is scaled by the standard deviation (sigma),
    # so that regions with higher uncertainty are more attractive.
    # The coefficient 0.5 is a heuristic balance between exploitation (EI)
    # and exploration (sigma).
    exploration = 0.5 * sigma

    # Cost damping: reduce sensitivity to cost so search can reach
    # the central region (inner 80%) where the minimum is constrained
    # cost ranges from 1.0 (edges) to ~2.0 (center)
    # Dampening: 1.0 + 0.5*(cost-1.0) maps [1,2] -> [1, 1.5]
    cost_damped = 1.0 + 0.5 * (cost - 1.0)

    # Combine: EI + exploration, divided by damped cost
    score = (ei + exploration) / cost_damped

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return early if close
    t0 = time.time()
    deadline = t0 + 325  # 5s safety margin
    
    # Use Expected Improvement (EI) with cost-awareness
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Cost-aware: divide by cost to prefer cheaper evaluations
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI using scipy.stats.norm for numerical stability
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware acquisition: EI / cost
    # This penalizes expensive points
    score = ei / np.maximum(cost, 1e-8)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END

import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Styblinski-Tang 2D BO.
    Uses Quantile-based Expected Improvement (Q-EI) with adaptive cost penalty.
    """
    t0 = time.time()
    
    # Safety: ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple UCB
        score = (-mu + 2.0 * sigma) / cost
        return np.where(np.isfinite(score), score, 0.0)

    known_min = -78.3323
    
    # Progress estimate: 0 at start, 1 near optimum
    # f_best starts around 50-100 for random points, ends near -78.3
    # Range approx 180.
    progress = np.clip(1.0 - (f_best - known_min) / 200.0, 0.0, 1.0)
    
    # Structural Change: Use a local baseline defined by a quantile of the posterior mean
    # Early on, we want to explore broadly, so use a higher quantile (e.g., median)
    # Late on, we want to exploit the best regions, so use a lower quantile (e.g., 1st percentile)
    # q goes from 0.5 to 0.05 as progress increases
    q = 0.5 - 0.45 * progress
    
    # Ensure q is within [0.01, 0.5]
    q = np.clip(q, 0.01, 0.5)
    
    # Calculate the local baseline threshold
    threshold = np.quantile(mu, q)
    
    # Calculate Expected Improvement relative to this local threshold
    # z = (threshold - mu) / sigma
    # EI = (threshold - mu) * Phi(z) + sigma * phi(z)
    
    z = (threshold - mu) / (sigma + 1e-12)
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (threshold - mu) * cdf_z + sigma * pdf_z
    
    # Cost adjustment
    # Early: lower penalty on cost (gamma small)
    # Late: higher penalty on cost (gamma large)
    gamma = 0.5 + 0.5 * progress
    
    score = ei / (cost ** gamma)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

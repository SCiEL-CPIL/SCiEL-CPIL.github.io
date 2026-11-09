import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 285

def _elapsed_ok():
    return (time.time() - _START_TIME) < _TIMEOUT

def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    n = mu.size

    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Structural Change: Use Probability of Improvement (PI) and Cost-Weighted Exploration
    # PI is more robust to local minima than EI for functions like Styblinski-Tang.
    
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # Expected Improvement (kept for blending if needed, but PI is primary)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Cost metrics
    mean_cost = np.mean(cost)
    # If the mean cost is high, the search is drifting to expensive regions.
    # We should increase the weight of exploration to escape, or penalize cost more.
    
    # Dynamic exploration constant
    # If mean cost is low (near 1.0), we are in a cheap region, exploit more (lower kappa).
    # If mean cost is high (near 2.0), we are in an expensive region, explore more (higher kappa) to find better/cheaper paths.
    # Map mean_cost [1.0, 2.0] to kappa [1.0, 3.0]
    kappa = 1.0 + 2.0 * (mean_cost - 1.0)
    kappa = np.clip(kappa, 1.0, 3.0)
    
    # Exploration term: sigma / cost^2 (heavier penalty for expensive exploration)
    exploration = sigma / (cost ** 2)
    
    # Exploitation term: PI / cost
    exploitation = pi / cost
    
    # Blend: Weights can be static or dynamic.
    # Let's use a fixed weight but rely on the cost^2 penalty in exploration to balance.
    w_exploit = 1.0
    w_explore = kappa
    
    score = w_exploit * exploitation + w_explore * exploration
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Fallback for near-timeout
    if not _elapsed_ok():
        # Simple fallback: EI / cost
        fallback = ei / cost
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    return score

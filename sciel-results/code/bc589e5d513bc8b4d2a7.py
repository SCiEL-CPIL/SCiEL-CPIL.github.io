import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
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

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    z = np.clip(z, -10, 10)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware EI with moderate exploitation bias
    # Reduced bias to 0.1 to balance exploration and exploitation,
    # preventing premature convergence to local minima in the multi-modal
    # Griewank landscape.
    exploitation_bias = -0.1 * mu

    # Dampen the cost penalty to avoid steering away from the central region
    # where the minimum is likely located (inner 80% box), as the cost is
    # higher there.
    cost_damped = 1.0 + 0.5 * (cost - 1.0)
    
    # Combine EI and exploitation bias
    score = (ei + exploitation_bias) / cost_damped

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score

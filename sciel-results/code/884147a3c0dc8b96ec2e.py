import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple score based on mu and cost
        cost_safe = np.maximum(cost, 1e-8)
        return np.nan_to_num((f_best - mu) / cost_safe, nan=0.0, posinf=0.0, neginf=0.0)

    xi = 1e-5  # Small improvement threshold
    sigma_safe = np.maximum(sigma, 1e-8)

    # z-score
    z = (f_best - xi - mu) / sigma_safe

    # Expected Improvement
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # Use stable norm.cdf and norm.pdf
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware score: multiplicative gate
    # score = EI * exp(-k * cost)
    # k is calibrated to the expected EI scale
    # cost is in [1, 2] (since x is in [-1,1]^8, scaled to [0,1], mean is in [0,1])
    # So cost ranges from 1 to 2
    # We want to suppress high-cost points unless EI is very large
    # Let's use k = 2.0, which gives exp(-2) ≈ 0.135 for cost=1 and exp(-4) ≈ 0.018 for cost=2
    # This creates a strong preference for low-cost points
    
    k = 2.0
    cost_gate = np.exp(-k * cost)
    
    scores = ei * cost_gate

    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

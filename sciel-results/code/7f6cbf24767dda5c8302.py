"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining adaptive LCB and EI per unit cost.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # Must finish within 300s

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive LCB: beta scales with relative uncertainty
    # Use a dynamic beta that encourages exploration when uncertainty is high
    # relative to the current best
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-10:
        # Scale beta: higher when sigma is large relative to f_best scale
        # f_best is negative for Hartmann 3D (min ~ -3.86), so use abs
        f_scale = max(abs(f_best), 1e-8)
        beta = 1.0 + 0.5 * (sigma_mean / f_scale)
    else:
        beta = 1.0
    
    # Lower Confidence Bound (higher is better for minimization, so we negate)
    # LCB = mu - beta * sigma; for maximization score, we use -LCB = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-normalized scores
    ei_cost = ei / cost_safe
    lcb_cost = lcb_score / cost_safe
    
    # Combine: weighted sum of EI and LCB per cost
    # Use a hybrid that balances exploitation (EI) and exploration (LCB)
    # The weight can be adjusted based on the uncertainty level
    # When sigma is high, favor LCB more; when sigma is low, favor EI more
    weight_lcb = np.clip(sigma_safe / (sigma_mean + 1e-8), 0.0, 1.0) * 0.5
    weight_ei = 1.0 - weight_lcb
    
    score = weight_ei * ei_cost + weight_lcb * lcb_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline - 5.0:
        # Return best available
        return score
    
    return score
# EVOLVE-BLOCK-END

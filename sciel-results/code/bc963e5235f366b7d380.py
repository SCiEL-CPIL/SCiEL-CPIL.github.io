"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost, with numerical safeguards.
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid overflow in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # Handle potential negative EI (shouldn't happen much, but clamp to 0)
    ei = np.maximum(ei, 0.0)
    # Divide by cost
    cost_safe = np.maximum(cost, 1e-12)
    score = ei / cost_safe
    # Ensure finite
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
# EVOLVE-BLOCK-END

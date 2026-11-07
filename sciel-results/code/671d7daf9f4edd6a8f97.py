"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variables for deadline guard
_start_time = None
_deadline = 300.0
_safety_margin = 5.0

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    if time.time() > _start_time + _deadline - _safety_margin:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-8)

    # Calculate z-score for Expected Improvement
    # mu, sigma, f_best are arrays
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    phi_z_f_best = norm.cdf(z)

    ei = (f_best - mu) * phi_z_f_best + sigma_safe * phi_z

    # Cost-aware score
    # cost is array of shape (2000,)
    # We want to maximize score
    # Score = EI / Cost
    # To allow for a bit more exploration in high uncertainty regions if EI is low,
    # we can add a small term, but EI already scales with sigma.
    # Let's stick to pure EI/Cost for stability, but ensure no division by zero.
    cost_safe = np.maximum(cost, 1e-8)

    # Use a power of cost to balance exploitation vs. exploration
    # A power of 0.5 is often more effective than 1.0 for cost-awareness
    # because it reduces the penalty for higher costs while still preferring cheaper points.
    cost_penalty = np.sqrt(cost_safe)

    # Base score: EI / cost_penalty
    scores = ei / cost_penalty

    # Add a small exploration bonus to encourage visiting unexplored regions
    # This helps when EI is low (e.g., when f_best is already good) but uncertainty is high
    # The bonus is scaled by the square root of sigma to be less aggressive than EI
    exploration_bonus = 0.1 * np.sqrt(sigma_safe) / cost_penalty

    scores = scores + exploration_bonus

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
